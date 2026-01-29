import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
import matplotlib.pyplot as plt
import ast

# Simple Neural Collaborative Filtering (DMF-like) with Content Features
# Chúng ta sẽ xây dựng một mô hình Neural Collaborative Filtering kết hợp Content (Hybrid)
# User Tower: Học từ User ID.
# Item Tower: Học từ Item ID + Genres (từ movies_cleaned.csv).

# Kiểm tra GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load dữ liệu
# Lưu ý: Điều chỉnh đường dẫn nếu cần thiết
train_path = 'notebooks/ratings_train.csv'
test_path = 'notebooks/ratings_test.csv'
movies_path = 'notebooks/movies_cleaned.csv'

# Đọc dữ liệu movies để lấy tên phim và genres
try:
    # Thêm cột genres
    df_movies = pd.read_csv(movies_path, usecols=['id', 'title', 'genres'], dtype={'id': str})
    # Chuyển đổi id sang numeric
    df_movies['id'] = pd.to_numeric(df_movies['id'], errors='coerce')
    df_movies = df_movies.dropna(subset=['id'])
    df_movies['id'] = df_movies['id'].astype(int)
    
    # Mapping ID -> Title
    movie_id_to_title = df_movies.set_index('id')['title'].to_dict()
    
    # Xử lý Genres: Parse string json -> list -> multi-hot vector
    print("Processing Genres...")
    df_movies['genres'] = df_movies['genres'].apply(
        lambda x: [d['name'] for d in ast.literal_eval(x)] if isinstance(x, str) else []
    )
    
    # Tạo mapping ID -> List Genres
    movie_id_to_genres = df_movies.set_index('id')['genres'].to_dict()
    
except Exception as e:
    print(f"Warning: Could not load/process movies file: {e}")
    movie_id_to_title = {}
    movie_id_to_genres = {}

# Đọc dữ liệu ratings (chỉ lấy các cột cần thiết để tiết kiệm bộ nhớ nếu file lớn)
cols = ['userId', 'movieId', 'rating']
try:
    df_train = pd.read_csv(train_path, usecols=cols)
    df_test = pd.read_csv(test_path, usecols=cols)
except FileNotFoundError:
    print("Files not found! Please check the paths.")
    exit()

print(f"Train size: {len(df_train)}")
print(f"Test size: {len(df_test)}")
print(df_train.head())

# Preprocessing: Encoding User and Movie IDs
# Chúng ta cần map userId và movieId về khoảng [0, N) để dùng trong Embedding Layer

# Gom tất cả user và item từ train và test để tạo mapping đầy đủ
all_users = pd.concat([df_train['userId'], df_test['userId']]).unique()
all_movies = pd.concat([df_train['movieId'], df_test['movieId']]).unique()

# Tạo encoder
user_encoder = LabelEncoder()
movie_encoder = LabelEncoder()

user_encoder.fit(all_users)
movie_encoder.fit(all_movies)

# Transform dữ liệu
df_train['user_idx'] = user_encoder.transform(df_train['userId'])
df_train['movie_idx'] = movie_encoder.transform(df_train['movieId'])

df_test['user_idx'] = user_encoder.transform(df_test['userId'])
df_test['movie_idx'] = movie_encoder.transform(df_test['movieId'])

num_users = len(all_users)
num_movies = len(all_movies)

print(f"Number of Users: {num_users}")
print(f"Number of Movies: {num_movies}")

# --- Content Feature Engineering ---
# 1. Tạo Encoder cho Genres (Multi-Hot)
# Lấy tất cả genres từ movie_id_to_genres nhưng phải đảm bảo thứ tự khớp với movie_encoder
# Chúng ta sẽ tạo một ma trận feature (Num_Movies x Num_Genres)
# Dòng i trong ma trận tương ứng với Movie có encoder_idx = i

# Lấy danh sách genres thực tế cho từng movie đã được encode
encoded_movie_genres = []
for idx in range(num_movies):
    real_movie_id = movie_encoder.inverse_transform([idx])[0]
    genres = movie_id_to_genres.get(real_movie_id, [])
    encoded_movie_genres.append(genres)

# MultiLabelBinarizer để chuyển list genres thành vector 0/1
mlb = MultiLabelBinarizer()
movie_features_matrix = mlb.fit_transform(encoded_movie_genres)
movie_features_tensor = torch.tensor(movie_features_matrix, dtype=torch.float32).to(device)
num_genres = movie_features_matrix.shape[1]

print(f"Content Features Shape: {movie_features_tensor.shape} (Num_Movies x Num_Genres)")
print(f"Example Genres: {mlb.classes_[:5]}")
# -----------------------------------

class RatingDataset(Dataset):
    def __init__(self, user_indices, movie_indices, ratings):
        self.users = torch.tensor(user_indices, dtype=torch.long)
        self.movies = torch.tensor(movie_indices, dtype=torch.long)
        self.ratings = torch.tensor(ratings, dtype=torch.float32)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.movies[idx], self.ratings[idx]

# Tạo Dataset và DataLoader
train_dataset = RatingDataset(df_train['user_idx'].values, df_train['movie_idx'].values, df_train['rating'].values)
test_dataset = RatingDataset(df_test['user_idx'].values, df_test['movie_idx'].values, df_test['rating'].values)

batch_size = 1024 # Batch size lớn giúp train nhanh hơn với dữ liệu lớn
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class SimpleDMF(nn.Module):
    def __init__(self, num_users, num_items, content_features, embedding_dim=64, hidden_dims=[64, 32]):
        super(SimpleDMF, self).__init__()
        
        self.movie_content = content_features      # (num_items, num_genres)
        num_content_features = content_features.shape[1]

        # User Embedding & MLP
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.user_layers = nn.ModuleList()
        input_dim = embedding_dim
        for dim in hidden_dims:
            self.user_layers.append(nn.Linear(input_dim, dim))
            self.user_layers.append(nn.ReLU())
            input_dim = dim
            
        # Item Embedding & MLP
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Item input là sự kết hợp của Embedding (ID) và Content Features (Genres)
        item_input_dim = embedding_dim + num_content_features
        
        self.item_layers = nn.ModuleList()
        input_dim = item_input_dim
        for dim in hidden_dims:
            self.item_layers.append(nn.Linear(input_dim, dim))
            self.item_layers.append(nn.ReLU())
            input_dim = dim
            
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)

    def forward(self, user_indices, item_indices):
        # User Tower
        user_vec = self.user_embedding(user_indices)
        for layer in self.user_layers:
            user_vec = layer(user_vec)
            
        # Item Tower
        # 1. Lấy ID Embedding
        item_emb = self.item_embedding(item_indices)
        
        # 2. Lấy Content Features (Genres) từ buffer
        content_vec = self.movie_content[item_indices]
        
        # 3. Nối (Concatenate) lại thành input cho MLP
        item_vec_combined = torch.cat([item_emb, content_vec], dim=1)
        
        item_vec = item_vec_combined
        for layer in self.item_layers:
            item_vec = layer(item_vec)
            
        # Interaction (Dot Product)
        interaction = (user_vec * item_vec).sum(dim=1)
        
        return interaction

# Khởi tạo mô hình
# Truyền thêm movie_features_tensor
model = SimpleDMF(num_users, num_movies, movie_features_tensor).to(device)
print(model)

# Loss và Optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
num_epochs = 10
train_losses = []

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    for users, movies, ratings in train_loader:
        users = users.to(device)
        movies = movies.to(device)
        ratings = ratings.to(device)
        
        optimizer.zero_grad()
        predictions = model(users, movies)
        loss = criterion(predictions, ratings)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * users.size(0)
        
    avg_loss = total_loss / len(train_dataset)
    train_losses.append(avg_loss)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

# Plot Loss (Sẽ không hiện plot trong script console, nhưng code vẫn giữ để đầy đủ)
# plt.plot(train_losses)
# plt.title('Training Loss')
# plt.xlabel('Epoch')
# plt.ylabel('MSE Loss')
# plt.show()

# Evaluation
model.eval()
predictions_list = []
targets_list = []

with torch.no_grad():
    for users, movies, ratings in test_loader:
        users = users.to(device)
        movies = movies.to(device)
        
        preds = model(users, movies)
        
        predictions_list.extend(preds.cpu().numpy())
        targets_list.extend(ratings.numpy())

rmse = np.sqrt(mean_squared_error(targets_list, predictions_list))
print(f"Test RMSE: {rmse:.4f}")

# Example Predictions
df_result = pd.DataFrame({'Actual': targets_list[:10], 'Predicted': predictions_list[:10]})

# Lấy lại movieId gốc để map tên phim
# test_dataset giữ nguyên thứ tự với test_loader (shuffle=False)
sample_users_idx = test_dataset.users[:10].numpy()
sample_movies_idx = test_dataset.movies[:10].numpy()

# Inverse transform để lấy ID gốc
sample_active_movie_ids = movie_encoder.inverse_transform(sample_movies_idx)

# Map tên phim
df_result['Movie Title'] = [movie_id_to_title.get(mid, f"Unknown ID: {mid}") for mid in sample_active_movie_ids]

print("\n--- Sample Predictions ---")
print(df_result)

# --- MRR@5 Calculation ---
# Tính MRR trên tập test (Re-ranking các item có trong tập test cho từng user)
# Lưu ý: Đây là đánh giá trên tập test có sẵn, không phải ranking trên toàn bộ item space (full ranking).
print("\nCalculating MRR@5...")

# Gán predictions vào dataframe test ban đầu
# Do test_loader có shuffle=False, thứ tự sẽ khớp nhau
df_test['predicted_rating'] = predictions_list

def calculate_mrr_at_k(df, k=5, threshold=4.0):
    mrr_sum = 0
    user_count = 0
    
    # Group theo User
    grouped = df.groupby('userId')
    
    for user_id, group in grouped:
        # Xác định các item "relevant" (User thực sự thích, ví dụ rating >= 4.0)
        relevant_items = group[group['rating'] >= threshold]
        
        # Nếu user không có item nào thích trong tập test, bỏ qua (hoặc coi như không đánh giá được)
        if relevant_items.empty:
            continue
            
        # Sắp xếp các item của user theo điểm dự đoán giảm dần (Ranking)
        sorted_group = group.sort_values(by='predicted_rating', ascending=False)
        
        # Lấy top k
        top_k = sorted_group.head(k)
        
        relevant_movie_ids = set(relevant_items['movieId'])
        
        reciprocal_rank = 0
        # Kiểm tra thứ hạng của item relevant đầu tiên tìm thấy
        for rank, row in enumerate(top_k.itertuples(), 1):
            if row.movieId in relevant_movie_ids:
                reciprocal_rank = 1.0 / rank
                break
        
        mrr_sum += reciprocal_rank
        user_count += 1
        
    if user_count == 0:
        return 0.0
    return mrr_sum / user_count

mrr5 = calculate_mrr_at_k(df_test, k=5, threshold=4.0)
print(f"Test MRR@5 (threshold=4.0): {mrr5:.4f}")

# Save Model
model_save_path = 'notebooks/dmf_model.pth'
torch.save(model.state_dict(), model_save_path)
print(f"\nModel saved to: {model_save_path}")
