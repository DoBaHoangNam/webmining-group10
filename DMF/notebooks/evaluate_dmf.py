import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
import ast

# --- 1. CONFIG & SETUP ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

train_path = 'notebooks/ratings_train.csv'
test_path = 'notebooks/ratings_test.csv'
movies_path = 'notebooks/movies_cleaned.csv'
model_path = 'notebooks/dmf_model.pth'

# --- 2. DATA LOADING & PREPROCESSING (Replicate training logic) ---
print("Loading data...")
cols = ['userId', 'movieId', 'rating']
df_train = pd.read_csv(train_path, usecols=cols)
df_test = pd.read_csv(test_path, usecols=cols)

# Load Movies & Genres
df_movies = pd.read_csv(movies_path, usecols=['id', 'title', 'genres'], dtype={'id': str})
df_movies['id'] = pd.to_numeric(df_movies['id'], errors='coerce')
df_movies = df_movies.dropna(subset=['id'])
df_movies['id'] = df_movies['id'].astype(int)

# Genre Processing
print("Processing content features...")
df_movies['genres'] = df_movies['genres'].apply(
    lambda x: [d['name'] for d in ast.literal_eval(x)] if isinstance(x, str) else []
)
movie_id_to_genres = df_movies.set_index('id')['genres'].to_dict()

# Encoders
print("Encoding IDs...")
all_users = pd.concat([df_train['userId'], df_test['userId']]).unique()
all_movies = pd.concat([df_train['movieId'], df_test['movieId']]).unique()

user_encoder = LabelEncoder()
movie_encoder = LabelEncoder()
user_encoder.fit(all_users)
movie_encoder.fit(all_movies)

# Feature Tensor Construction
num_movies = len(all_movies)
encoded_movie_genres = []
for idx in range(num_movies):
    real_movie_id = movie_encoder.inverse_transform([idx])[0]
    genres = movie_id_to_genres.get(real_movie_id, [])
    encoded_movie_genres.append(genres)

mlb = MultiLabelBinarizer()
movie_features_matrix = mlb.fit_transform(encoded_movie_genres)
movie_features_tensor = torch.tensor(movie_features_matrix, dtype=torch.float32).to(device)

# Prepare Test Data
df_test['user_idx'] = user_encoder.transform(df_test['userId'])
df_test['movie_idx'] = movie_encoder.transform(df_test['movieId'])

class RatingDataset(Dataset):
    def __init__(self, user_indices, movie_indices, ratings):
        self.users = torch.tensor(user_indices, dtype=torch.long)
        self.movies = torch.tensor(movie_indices, dtype=torch.long)
        self.ratings = torch.tensor(ratings, dtype=torch.float32)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.movies[idx], self.ratings[idx]

test_dataset = RatingDataset(df_test['user_idx'].values, df_test['movie_idx'].values, df_test['rating'].values)
test_loader = DataLoader(test_dataset, batch_size=4096, shuffle=False)

# --- 3. DEFINE MODEL (Must match training structure) ---
class SimpleDMF(nn.Module):
    def __init__(self, num_users, num_items, content_features, embedding_dim=64, hidden_dims=[64, 32]):
        super(SimpleDMF, self).__init__()
        self.movie_content = content_features 
        num_content_features = content_features.shape[1]

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.user_layers = nn.ModuleList()
        input_dim = embedding_dim
        for dim in hidden_dims:
            self.user_layers.append(nn.Linear(input_dim, dim))
            self.user_layers.append(nn.ReLU())
            input_dim = dim
            
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        item_input_dim = embedding_dim + num_content_features
        self.item_layers = nn.ModuleList()
        input_dim = item_input_dim
        for dim in hidden_dims:
            self.item_layers.append(nn.Linear(input_dim, dim))
            self.item_layers.append(nn.ReLU())
            input_dim = dim
            
    def forward(self, user_indices, item_indices):
        user_vec = self.user_embedding(user_indices)
        for layer in self.user_layers:
            user_vec = layer(user_vec)
            
        item_emb = self.item_embedding(item_indices)
        content_vec = self.movie_content[item_indices]
        item_vec = torch.cat([item_emb, content_vec], dim=1)
        for layer in self.item_layers:
            item_vec = layer(item_vec)
            
        interaction = (user_vec * item_vec).sum(dim=1)
        return interaction

# --- 4. LOAD MODEL ---
print("Loading model weights...")
num_users = len(all_users)
model = SimpleDMF(num_users, num_movies, movie_features_tensor).to(device)
try:
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"Model file not found at {model_path}. Please train the model first.")
    exit()

# --- 5. PREDICTION & METRICS CALCULATION ---
print("Generating predictions on test set...")
predictions_list = []
with torch.no_grad():
    for users, movies, ratings in test_loader:
        users = users.to(device)
        movies = movies.to(device)
        preds = model(users, movies)
        predictions_list.extend(preds.cpu().numpy())

df_test['predicted_rating'] = predictions_list

# Metric Functions
def calculate_mse(df):
    return mean_squared_error(df['rating'], df['predicted_rating'])

def calculate_ranking_metrics(df, k=5, threshold=3.5):
    mrr_sum = 0
    hr_sum = 0
    user_count = 0
    
    grouped = df.groupby('userId')
    
    for _, group in grouped:
        relevant_items = group[group['rating'] >= threshold]
        if relevant_items.empty:
            continue
            
        sorted_group = group.sort_values(by='predicted_rating', ascending=False)
        top_k = sorted_group.head(k)
        relevant_ids = set(relevant_items['movieId'])
        
        # MRR
        reciprocal_rank = 0
        for rank, row in enumerate(top_k.itertuples(), 1):
            if row.movieId in relevant_ids:
                reciprocal_rank = 1.0 / rank
                break
        mrr_sum += reciprocal_rank
        
        # HR (Hit Rate): At least one relevant item in top K
        # Check intersection between top_k movieIds and relevant_ids
        top_k_ids = set(top_k['movieId'])
        if not top_k_ids.isdisjoint(relevant_ids):
            hr_sum += 1
            
        user_count += 1
        
    mrr = mrr_sum / user_count if user_count > 0 else 0.0
    hr = hr_sum / user_count if user_count > 0 else 0.0
    return mrr, hr

# --- 6. IDENTIFY USER GROUPS ---
print("Identifying user groups...")
# Calculate rating counts based on TRAIN set
user_counts = df_train['userId'].value_counts()

# Top 5 most active
top_5_users = user_counts.head(5).index.tolist()

# Bottom 5 least active (must appear in test set to evaluate)
# Get users who are in Test set
users_in_test = set(df_test['userId'].unique())
# Filter counts to only include those in test
test_user_counts = user_counts[user_counts.index.isin(users_in_test)]
bottom_5_users = test_user_counts.nsmallest(5).index.tolist()

print(f"Top 5 Users (Most Active): {top_5_users}")
print(f"Bottom 5 Users (Least Active in Test): {bottom_5_users}")

# --- 7. REPORT ---
print("\n" + "="*40)
print("             EVALUATION REPORT            ")
print("="*40)

# 1. Global Metrics
mse_global = calculate_mse(df_test)
mrr_global, hr_global = calculate_ranking_metrics(df_test, k=5)

print(f"Global MSE:   {mse_global:.4f}")
print(f"Global HR@5:  {hr_global:.4f}")
print(f"Global MRR@5: {mrr_global:.4f}")
print("-" * 40)

# 2. Top 5 Most Active
df_top = df_test[df_test['userId'].isin(top_5_users)]
if not df_top.empty:
    mse_top = calculate_mse(df_top)
    mrr_top, hr_top = calculate_ranking_metrics(df_top, k=5)
    print(f"Top 5 Active Users - Mean MSE:   {mse_top:.4f}")
    print(f"Top 5 Active Users - Mean HR@5:  {hr_top:.4f}")
    print(f"Top 5 Active Users - Mean MRR@5: {mrr_top:.4f}")
else:
    print("No data in test set for Top 5 Active Users.")

print("-" * 40)

# 3. Bottom 5 Least Active
df_bottom = df_test[df_test['userId'].isin(bottom_5_users)]
if not df_bottom.empty:
    mse_bottom = calculate_mse(df_bottom)
    mrr_bottom, hr_bottom = calculate_ranking_metrics(df_bottom, k=5)
    print(f"Top 5 Least Users - Mean MSE:    {mse_bottom:.4f}")
    print(f"Top 5 Least Users - Mean HR@5:   {hr_bottom:.4f}")
    print(f"Top 5 Least Users - Mean MRR@5:  {mrr_bottom:.4f}")
else:
    print("No data in test set for Bottom 5 Least Active Users.")
print("="*40)
