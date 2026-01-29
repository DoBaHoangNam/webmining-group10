# lightgcn_bpr_train.py
# pip install pandas numpy torch scipy

import math
import random
import numpy as np
import pandas as pd
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import coo_matrix

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# 1) Config
# =========================
TRAIN_PATH = "train.csv"
TEST_PATH  = "test.csv"

USER_COL = "userId"
ITEM_COL = "movieId"
RATING_COL = "rating"

# Rating threshold để xác định positive interactions (MovieLens hay dùng >=4)
POS_THRESHOLD = 3.5

EMB_DIM = 64
K_LAYERS = 2
LR = 1e-3
WEIGHT_DECAY = 1e-6
BATCH_SIZE = 2048
EPOCHS = 30

# eval
TOPK = 5
EVAL_EVERY = 1  # epoch

# =========================
# 2) Data utilities
# =========================
def load_interactions(path: str):
    df = pd.read_csv(path)
    # chỉ lấy cột cần thiết
    df = df[[USER_COL, ITEM_COL, RATING_COL]].copy()
    # positive filter
    df = df[df[RATING_COL] >= POS_THRESHOLD]
    df = df.drop_duplicates(subset=[USER_COL, ITEM_COL])
    return df

def build_id_maps(train_df, test_df):
    # đảm bảo mapping phủ cả user/item xuất hiện ở train/test
    all_users = pd.concat([train_df[USER_COL], test_df[USER_COL]]).unique()
    all_items = pd.concat([train_df[ITEM_COL], test_df[ITEM_COL]]).unique()

    user2idx = {u:i for i,u in enumerate(all_users)}
    item2idx = {m:i for i,m in enumerate(all_items)}

    idx2user = {i:u for u,i in user2idx.items()}
    idx2item = {i:m for m,i in item2idx.items()}
    return user2idx, item2idx, idx2user, idx2item

def df_to_index_pairs(df, user2idx, item2idx):
    u = df[USER_COL].map(user2idx).astype(int).values
    i = df[ITEM_COL].map(item2idx).astype(int).values
    return u, i

def build_user_pos_dict(u_idx, i_idx, n_users):
    user_pos = [[] for _ in range(n_users)]
    user_pos_set = [set() for _ in range(n_users)]
    for u, it in zip(u_idx, i_idx):
        user_pos[u].append(it)
        user_pos_set[u].add(it)
    return user_pos, user_pos_set

# =========================
# 3) Graph building (A_norm)
# =========================
def build_normalized_adj(n_users, n_items, u_idx, i_idx):
    """
    Build symmetric normalized adjacency for bipartite graph:
    nodes = [users (0..U-1), items (U..U+I-1)]
    edges: (u, U+i) and (U+i, u)
    A_norm = D^{-1/2} A D^{-1/2}
    """
    num_nodes = n_users + n_items

    # bipartite edges (undirected)
    rows = np.concatenate([u_idx, i_idx + n_users])
    cols = np.concatenate([i_idx + n_users, u_idx])
    data = np.ones_like(rows, dtype=np.float32)

    A = coo_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))

    # degree
    deg = np.array(A.sum(axis=1)).squeeze()
    deg_inv_sqrt = np.power(deg, -0.5, where=deg>0)
    deg_inv_sqrt[deg == 0] = 0.0

    # normalized values: A_ij * d_i^{-1/2} * d_j^{-1/2}
    norm_data = data * deg_inv_sqrt[rows] * deg_inv_sqrt[cols]

    A_norm = coo_matrix((norm_data, (rows, cols)), shape=(num_nodes, num_nodes))
    return A_norm

def scipy_coo_to_torch_sparse(coo: coo_matrix):
    coo = coo.tocoo()
    indices = torch.tensor(np.vstack((coo.row, coo.col)), dtype=torch.long)
    values  = torch.tensor(coo.data, dtype=torch.float32)
    shape   = torch.Size(coo.shape)
    return torch.sparse_coo_tensor(indices, values, shape).coalesce()

# =========================
# 4) LightGCN model
# =========================
class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, emb_dim, k_layers, A_norm_torch):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.k_layers = k_layers

        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)

        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)

        # fixed normalized adjacency
        self.A_norm = A_norm_torch  # torch sparse [U+I, U+I]

    def propagate(self):
        """
        Returns final embeddings z_user, z_item after layer combination:
        z = mean_{k=0..K} e^{(k)}
        """
        # e^(0)
        e0_users = self.user_emb.weight
        e0_items = self.item_emb.weight
        e = torch.cat([e0_users, e0_items], dim=0)  # [U+I, d]

        all_layers = [e]

        for _ in range(self.k_layers):
            e = torch.sparse.mm(self.A_norm, e)  # message passing
            all_layers.append(e)

        # layer combination (mean)
        z = torch.stack(all_layers, dim=0).mean(dim=0)  # [U+I, d]

        z_users = z[:self.n_users]
        z_items = z[self.n_users:]
        return z_users, z_items

    def score(self, z_users, z_items, u, i):
        # dot product
        return (z_users[u] * z_items[i]).sum(dim=-1)

# =========================
# 5) BPR training helpers
# =========================
def sample_batch(user_pos, user_pos_set, n_users, n_items, batch_size):
    """
    Samples (u, pos, neg) triples.
    Pick u that has at least 1 positive.
    pos from user_pos[u], neg random not in user_pos_set[u].
    """
    users = []
    pos_items = []
    neg_items = []

    while len(users) < batch_size:
        u = random.randrange(n_users)
        if not user_pos[u]:
            continue
        pos = random.choice(user_pos[u])

        # negative sampling
        while True:
            neg = random.randrange(n_items)
            if neg not in user_pos_set[u]:
                break

        users.append(u)
        pos_items.append(pos)
        neg_items.append(neg)

    return (
        torch.tensor(users, dtype=torch.long, device=DEVICE),
        torch.tensor(pos_items, dtype=torch.long, device=DEVICE),
        torch.tensor(neg_items, dtype=torch.long, device=DEVICE),
    )

def bpr_loss(model, z_users, z_items, u, pos, neg, reg_lambda=1e-4):
    s_pos = model.score(z_users, z_items, u, pos)
    s_neg = model.score(z_users, z_items, u, neg)

    loss = -torch.log(torch.sigmoid(s_pos - s_neg) + 1e-12).mean()

    # L2 reg on involved embeddings (classic)
    reg = (model.user_emb(u).norm(2).pow(2) +
           model.item_emb(pos).norm(2).pow(2) +
           model.item_emb(neg).norm(2).pow(2)) / u.shape[0]
    return loss + reg_lambda * reg

# =========================
# 6) Evaluation: Recall@K, NDCG@K
# =========================
@torch.no_grad()
def eval_topk(model, train_user_pos_set, test_user_gt, n_items, k=5):
    """
    test_user_gt: dict u -> set(items) (ground truth positives in test)
    train_user_pos_set: list[set] of train positives (to filter)
    """
    model.eval()
    z_users, z_items = model.propagate()
    z_users = z_users.to(DEVICE)
    z_items = z_items.to(DEVICE)

    users = list(test_user_gt.keys())
    if not users:
        return {"recall": 0.0, "ndcg": 0.0}

    # batch compute scores
    user_tensor = torch.tensor(users, dtype=torch.long, device=DEVICE)
    user_emb = z_users[user_tensor]  # [B, d]
    scores = user_emb @ z_items.t()  # [B, I]

    # filter seen items in train
    for row_idx, u in enumerate(users):
        seen = train_user_pos_set[u]
        if seen:
            scores[row_idx, torch.tensor(list(seen), device=DEVICE)] = -1e9

    topk_scores, topk_items = torch.topk(scores, k=k, dim=1)

    recalls = []
    ndcgs = []

    for row_idx, u in enumerate(users):
        gt = test_user_gt[u]
        if not gt:
            continue

        rec_list = topk_items[row_idx].tolist()

        # Recall@K
        hit = sum(1 for it in rec_list if it in gt)
        recalls.append(hit / len(gt))

        # NDCG@K
        dcg = 0.0
        for rank, it in enumerate(rec_list, start=1):
            if it in gt:
                dcg += 1.0 / math.log2(rank + 1)

        # IDCG
        ideal_hits = min(len(gt), k)
        idcg = sum(1.0 / math.log2(r + 1) for r in range(1, ideal_hits + 1))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

    return {
        "recall": float(np.mean(recalls) if recalls else 0.0),
        "ndcg": float(np.mean(ndcgs) if ndcgs else 0.0),
    }

# =========================
# 7) Main
# =========================
def main():
    train_df = load_interactions(TRAIN_PATH)
    test_df  = load_interactions(TEST_PATH)

    user2idx, item2idx, idx2user, idx2item = build_id_maps(train_df, test_df)

    n_users = len(user2idx)
    n_items = len(item2idx)

    train_u, train_i = df_to_index_pairs(train_df, user2idx, item2idx)
    test_u,  test_i  = df_to_index_pairs(test_df,  user2idx, item2idx)

    user_pos, train_user_pos_set = build_user_pos_dict(train_u, train_i, n_users)

    # build test ground truth per user
    test_user_gt = defaultdict(set)
    for u, it in zip(test_u, test_i):
        test_user_gt[u].add(it)

    # build normalized adjacency from TRAIN ONLY
    A_norm = build_normalized_adj(n_users, n_items, train_u, train_i)
    A_norm_torch = scipy_coo_to_torch_sparse(A_norm).to(DEVICE)

    model = LightGCN(n_users, n_items, EMB_DIM, K_LAYERS, A_norm_torch).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # number of iterations per epoch (heuristic)
    num_train_edges = len(train_u)
    iters_per_epoch = max(1, num_train_edges // BATCH_SIZE)

    print(f"Users={n_users}, Items={n_items}, TrainPosEdges={num_train_edges}, TestUsersWithGT={len(test_user_gt)}")
    print(f"DEVICE={DEVICE}, iters/epoch={iters_per_epoch}")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for _ in range(iters_per_epoch):
            u, pos, neg = sample_batch(user_pos, train_user_pos_set, n_users, n_items, BATCH_SIZE)

            z_users, z_items = model.propagate()
            loss = bpr_loss(model, z_users, z_items, u, pos, neg, reg_lambda=1e-4)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / iters_per_epoch

        if epoch % EVAL_EVERY == 0:
            metrics = eval_topk(model, train_user_pos_set, test_user_gt, n_items, k=TOPK)
            print(f"[Epoch {epoch:03d}] loss={avg_loss:.4f} | Recall@{TOPK}={metrics['recall']:.4f} | NDCG@{TOPK}={metrics['ndcg']:.4f}")
        else:
            print(f"[Epoch {epoch:03d}] loss={avg_loss:.4f}")

    # Example: recommend top-5 for a raw userId
    @torch.no_grad()
    def recommend_for_user(raw_user_id, topk=5):
        if raw_user_id not in user2idx:
            print("User not found in mapping.")
            return []

        u = user2idx[raw_user_id]
        model.eval()
        z_users, z_items = model.propagate()

        scores = z_users[u:u+1] @ z_items.t()   # [1, I]
        seen = train_user_pos_set[u]
        if seen:
            scores[0, torch.tensor(list(seen), device=DEVICE)] = -1e9

        _, top_items = torch.topk(scores, k=topk, dim=1)
        top_items = top_items[0].tolist()
        # map back to movieId
        return [idx2item[i] for i in top_items]

    # demo
    some_user = next(iter(user2idx.keys()))
    print("Demo recommend for userId =", some_user, "->", recommend_for_user(some_user, topk=5))

if __name__ == "__main__":
    main()
