import os
from typer import Typer
from utils import write_jsonl

app = Typer()

@app.command()
def placeholder():
    """A placeholder command that does nothing."""
    pass

@app.command()
def extract_user_data(data_path:str='data/only_ratings_train.csv'):
    """Extract user data including interactions, negative movies, and possible negative movies."""
    import pandas as pd
    
    # Read the dataset
    df = pd.read_csv(data_path)
    
    # Get all movies that have ratings < 2.5 (disliked movies across all users)
    disliked_movies = df[df['rating'] < 2.5]['movieId'].unique()
    
    # Group by user and extract data
    user_data = {}
    for user_id in df['userId'].unique():
        user_df = df[df['userId'] == user_id]
        
        # Extract interactions (ratings and timestamps for all movies)
        interactions = user_df[['movieId', 'rating', 'timestamp']].to_dict('records')
        
        # Extract negative movies (movies user watched with rating < 2.5)
        negative_movies = user_df[user_df['rating'] < 2.5]['movieId'].tolist()
        
        # Get movies watched by this user
        watched_movies = set(user_df['movieId'].unique())
        
        # Possible negative movies: disliked by others but not watched by this user
        possible_negative_movies = [mid for mid in disliked_movies if mid not in watched_movies]
        
        user_data[user_id] = {
            'interactions': interactions,
            'negative_movies': negative_movies,
            'possible_negative_movies': possible_negative_movies
        }
    
    print(f"Extracted data for {len(user_data)} users")
    return user_data

@app.command()
def reindex_user_item(data_path:str='data/ratings_train_clean.csv', save_path:str='data/only_ratings_train.csv', mapping_path:str='data/index_mappings.json'):
    """Reindex user and item IDs to sequential indices and save mappings."""
    import pandas as pd
    import json
    from pathlib import Path
    
    # Read the dataset
    df = pd.read_csv(data_path)
    df = df[['userId', 'movieId', 'rating', 'timestamp']]
    
    # Create mappings: old ID -> new index (0-based sequential)
    unique_users = sorted(df['userId'].unique())
    unique_movies = sorted(df['movieId'].unique())
    
    user_old_to_new = {int(old_id): new_idx + 1  for new_idx, old_id in enumerate(unique_users)}
    movie_old_to_new = {int(old_id): new_idx + 1 for new_idx, old_id in enumerate(unique_movies)}
    
    # Create reverse mappings: new index -> old ID
    user_new_to_old = {new_idx: int(old_id) for old_id, new_idx in user_old_to_new.items()}
    movie_new_to_old = {new_idx: int(old_id) for old_id, new_idx in movie_old_to_new.items()}
    
    # Apply reindexing to dataframe
    df_reindexed = df.copy()
    df_reindexed['userId'] = df_reindexed['userId'].map(user_old_to_new)
    df_reindexed['movieId'] = df_reindexed['movieId'].map(movie_old_to_new)
    
    # Save reindexed data
    reindexed_path = save_path
    df_reindexed.to_csv(reindexed_path, index=False)
    print(f"Saved reindexed data to: {reindexed_path}")
    
    # Save both mappings to a single file
    mapping_path = mapping_path
    with open(mapping_path, 'w') as f:
        json.dump({
            'user_mapping': {
                'old_to_new': user_old_to_new,
                'new_to_old': user_new_to_old
            },
            'movie_mapping': {
                'old_to_new': movie_old_to_new,
                'new_to_old': movie_new_to_old
            }
        }, f, indent=2)
    print(f"Saved mappings to: {mapping_path}")
    
    print(f"\nReindexing summary:")
    print(f"  Users: {len(unique_users)} (0 to {len(unique_users)-1})")
    print(f"  Movies: {len(unique_movies)} (0 to {len(unique_movies)-1})")
    print(f"  Total ratings: {len(df_reindexed)}")
    
    return {
        'reindexed_data': df_reindexed,
        'user_mapping': {'old_to_new': user_old_to_new, 'new_to_old': user_new_to_old},
        'movie_mapping': {'old_to_new': movie_old_to_new, 'new_to_old': movie_new_to_old}
    }

@app.command()
def preprocess_data(input_path:str='data/ratings_test_clean.csv', output_path:str='data/only_ratings_test.csv', index_mapping_path:str='data/index_mappings.json'):
    """Read data and reindex using existing mappings, keeping only relevant columns."""
    import pandas as pd
    import json
    
    # Read the dataset
    print(f"Reading data from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"Original data shape: {df.shape}")
    
    # Keep only required columns
    required_columns = ['userId', 'movieId', 'rating', 'timestamp']
    df = df[required_columns]
    print(f"After selecting columns: {df.shape}")
    
    # Load index mappings
    print(f"Loading index mappings from: {index_mapping_path}")
    with open(index_mapping_path, 'r') as f:
        mappings = json.load(f)
    
    # Extract old_to_new mappings (keys are strings in JSON, convert to int)
    user_mapping = {int(k): v for k, v in mappings['user_mapping']['old_to_new'].items()}
    movie_mapping = {int(k): v for k, v in mappings['movie_mapping']['old_to_new'].items()}
    
    print(f"User mapping size: {len(user_mapping)}")
    print(f"Movie mapping size: {len(movie_mapping)}")
    
    # Filter out users and items not in mapping
    df_filtered = df[
        df['userId'].isin(user_mapping.keys()) & 
        df['movieId'].isin(movie_mapping.keys())
    ].copy()
    
    dropped_rows = len(df) - len(df_filtered)
    print(f"Dropped {dropped_rows} rows with unmapped users/items")
    print(f"After filtering: {df_filtered.shape}")
    
    # Apply reindexing
    df_filtered['userId'] = df_filtered['userId'].map(user_mapping)
    df_filtered['movieId'] = df_filtered['movieId'].map(movie_mapping)
    
    # Save preprocessed data
    df_filtered.to_csv(output_path, index=False)
    print(f"Saved preprocessed data to: {output_path}")
    print(f"\nFinal data shape: {df_filtered.shape}")
    print(f"Users: {df_filtered['userId'].nunique()}")
    print(f"Movies: {df_filtered['movieId'].nunique()}")
    
    return df_filtered

@app.command()
def extract_interaction_history_for_training(
    input_file:str='data/only_ratings_train.csv',
    output_file:str='data/exp1/train.jsonl',
    max_history_length:int=50,
    min_positive_threshold:float=3.5,
    do_augment_interactions:bool=True,
    augmentation_type:str='nearest_positive',# 'nearest_positive' or 'random_positive'
    num_negative_samples:int=50
):
    """Extract interaction sequences for training a sequential recommender system.
    
    For each user:
    - Creates sequences from positive items (rating >= min_positive_threshold)
    - Each sequence has: user_id, interaction_history (items before target), target_item_id, negative_samples
    - If do_augment_interactions=True, also creates sequences using negative items as pivot points
    """
    assert augmentation_type in ['nearest_positive', 'random_positive'], "augmentation_type must be 'nearest_positive' or 'random_positive'"
    import pandas as pd
    import json
    import random
    from pathlib import Path
    
    # Read data
    print(f"Reading data from: {input_file}")
    df = pd.read_csv(input_file)
    
    # Sort by userId and timestamp to ensure chronological order
    df = df.sort_values(['userId', 'timestamp']).reset_index(drop=True)
    
    # Get all unique items in the dataset
    all_items = set(int(x) for x in df['movieId'].unique())
    
    sequences = []
    
    # Process each user
    for user_id in df['userId'].unique():
        user_df = df[df['userId'] == user_id].reset_index(drop=True)
        
        # Get user's interactions in chronological order
        interactions = user_df[['movieId', 'rating', 'timestamp']].values
        
        # Identify positive and negative item indices for this user
        positive_indices = [i for i, (mid, rating, ts) in enumerate(interactions) 
                           if rating >= min_positive_threshold]
        negative_indices = [i for i, (mid, rating, ts) in enumerate(interactions) 
                           if rating < min_positive_threshold]
        
        # Get user's watched items (all interactions regardless of rating)
        user_watched_items = set(int(x) for x in user_df['movieId'].values)
        
        # Negative candidates: all items the user has never watched
        # This matches the testing strategy for consistency
        unwatched_items = list(all_items - user_watched_items)
        
        # Get user's negative items (low-rated by this user)
        user_negative_items = [int(x) for x in user_df[user_df['rating'] < min_positive_threshold]['movieId'].values]
        
        # Combine unwatched items with user's negative items for sampling
        # This ensures negatives include both movies never watched AND movies user disliked
        all_neg_candidates = list(set(unwatched_items + user_negative_items))
        
        if not all_neg_candidates:
            continue  # Skip users with no negative samples available
        
        # 1. Sample normal sequences from positive items
        for pos_idx in positive_indices:
            target_item_id = int(interactions[pos_idx][0])
            target_timestamp = int(interactions[pos_idx][2])
            
            # Get history before this positive item (up to max_history_length items)
            start_idx = max(0, pos_idx - max_history_length)
            history_items = [int(interactions[i][0]) for i in range(start_idx, pos_idx)]
            if len(history_items) == 0:
                continue  # Skip if no history
            
            # Sample negative items for this sequence
            neg_samples = random.sample(all_neg_candidates, 
                                       min(num_negative_samples, len(all_neg_candidates)))
            
            # Add timestamp information
            sequences.append({
                'user_id': int(user_id),
                'interaction_history': history_items,
                'target_item_id': target_item_id,
                'target_timestamp': target_timestamp,
                'negative_samples': neg_samples
            })
        
        # 2. Augmented sequences using negative items as pivot points
        if do_augment_interactions:
            for neg_idx in negative_indices:
                # Get history up to (but not including) this negative item
                start_idx = max(0, neg_idx - max_history_length)
                history_items = [int(interactions[i][0]) for i in range(start_idx, neg_idx)]
                if len(history_items) == 0:
                    continue  # Skip if no history
                # Find future positive items after this negative item
                future_positive_indices = [i for i in positive_indices if i > neg_idx]
                
                if not future_positive_indices:
                    continue  # No future positive items, skip this augmentation
                
                # Select target based on augmentation type
                if augmentation_type == 'nearest_positive':
                    target_idx = future_positive_indices[0]
                else:  # random_positive
                    target_idx = random.choice(future_positive_indices)
                
                target_item_id = int(interactions[target_idx][0])
                target_timestamp = int(interactions[neg_idx][2])
                
                # Sample negative items for this sequence
                neg_samples = random.sample(all_neg_candidates, 
                                           min(num_negative_samples, len(all_neg_candidates)))
                
                # Add timestamp information
                sequences.append({
                    'user_id': int(user_id),
                    'interaction_history': history_items,
                    'target_item_id': target_item_id,
                    'target_timestamp': target_timestamp,
                    'negative_samples': neg_samples
                })
    
    # Save to JSON
    os.makedirs(Path(output_file).parent, exist_ok=True)
    write_jsonl(output_file, sequences, mode='w')

    print(f"\nExtracted {len(sequences)} training sequences")
    print(f"  Users processed: {df['userId'].nunique()}")
    print(f"  Augmentation: {'enabled' if do_augment_interactions else 'disabled'}")
    print(f"Saved to: {output_file}")
    
    return sequences

@app.command()
def extract_interaction_history_for_evaluation(
    input_file:str='data/only_ratings_test.csv',
    train_history_file:str='data/only_ratings_train.csv',
    output_file:str='data/exp1/test.jsonl',
    max_history_length:int=50,
    min_positive_threshold:float=3.5,
    num_negative_samples:int=50):
    """Extract interaction sequences for evaluating a sequential recommender system.
    
    For each user in test set:
    - Creates sequences only from positive items (rating >= min_positive_threshold)
    - History may include interactions from both train and test sets
    - Negative samples are movies the user has never watched
    """
    import pandas as pd
    import json
    import random
    import os
    from pathlib import Path
    from collections import defaultdict
    
    # Read test data
    print(f"Reading test data from: {input_file}")
    df_test = pd.read_csv(input_file)
    df_test = df_test.sort_values(['userId', 'timestamp']).reset_index(drop=True)
    
    # Read training data to get interaction history per user
    print(f"Reading training data from: {train_history_file}")
    df_train = pd.read_csv(train_history_file)
    df_train = df_train.sort_values(['userId', 'timestamp']).reset_index(drop=True)
    
    # For each user, get their complete training interaction history
    user_train_history = {}
    for user_id in df_train['userId'].unique():
        user_train_df = df_train[df_train['userId'] == user_id]
        # Get all movie IDs from training in chronological order
        train_items = [int(x) for x in user_train_df['movieId'].values]
        user_train_history[int(user_id)] = train_items
    
    print(f"Found training history for {len(user_train_history)} users")
    
    # Get all unique items across both train and test datasets
    all_items = set(int(x) for x in df_test['movieId'].unique())
    all_items.update(int(x) for x in df_train['movieId'].unique())
    
    sequences = []
    
    # Process each user in test set
    for user_id in df_test['userId'].unique():
        user_df = df_test[df_test['userId'] == user_id].reset_index(drop=True)
        
        # Get user's test interactions in chronological order
        test_interactions = user_df[['movieId', 'rating', 'timestamp']].values
        
        # Identify positive item indices (rating >= threshold)
        positive_indices = [i for i, (mid, rating, ts) in enumerate(test_interactions) 
                           if rating >= min_positive_threshold]
        
        if not positive_indices:
            continue  # Skip users with no positive test items
        
        # Get user's watched items (from test set)
        user_watched_items = set(int(x) for x in user_df['movieId'].values)
        
        # Get user's train history if available
        train_history = user_train_history.get(int(user_id), [])
        
        # Add train history to watched items
        user_watched_items.update(train_history)
        
        # Negative samples: items never watched by user
        unwatched_items = list(all_items - user_watched_items)
        
        # Create sequences for each positive test item
        for pos_idx in positive_indices:
            target_item_id = int(test_interactions[pos_idx][0])
            target_timestamp = int(test_interactions[pos_idx][2])
            
            # Build history: train history + test history up to (but not including) target
            test_history = [int(test_interactions[i][0]) for i in range(pos_idx)]
            combined_history = train_history + test_history
            
            # Limit to max_history_length (take most recent items)
            if len(combined_history) > max_history_length:
                combined_history = combined_history[-max_history_length:]
            
            if len(combined_history) == 0:
                continue  # Skip if no history available
            
            # Get negative items from this timestamp onward in test set
            future_negative_items = [
                int(test_interactions[i][0]) 
                for i in range(pos_idx, len(test_interactions))
                if test_interactions[i][1] < min_positive_threshold
            ]
            
            # Combine unwatched items with future negative items
            negative_candidates = list(set(unwatched_items + future_negative_items))
            
            if not negative_candidates:
                continue  # Skip if no negative candidates
            
            # Sample negative items
            neg_samples = random.sample(negative_candidates, 
                                       min(num_negative_samples, len(negative_candidates)))
            
            sequences.append({
                'user_id': int(user_id),
                'interaction_history': combined_history,
                'target_item_id': target_item_id,
                'target_timestamp': target_timestamp,
                'negative_samples': neg_samples
            })
    
    # Save to JSONL
    os.makedirs(Path(output_file).parent, exist_ok=True)
    write_jsonl(output_file, sequences, mode='w')
    
    print(f"\nExtracted {len(sequences)} evaluation sequences")
    print(f"  Users processed: {df_test['userId'].nunique()}")
    print(f"  Users with sequences: {len(set(seq['user_id'] for seq in sequences))}")
    print(f"Saved to: {output_file}")
    
    return sequences


@app.command()
def split_train_val(input_path: str='data/exp1/train.jsonl', train_output_path: str='data/exp1/train_split.jsonl', val_output_path: str='data/exp1/val_split.jsonl', train_ratio: float=0.9):
    """Split the extracted training sequences into training and validation sets."""
    from utils import read_jsonl, write_jsonl
    import random
    import os
    from pathlib import Path
    from collections import defaultdict
    
    # Read sequences
    print(f"Reading sequences from: {input_path}")
    sequences = read_jsonl(input_path)
    print(f"Total sequences: {len(sequences)}")
    
    # Group sequences by user_id
    user_sequences = defaultdict(list)
    for seq in sequences:
        user_sequences[seq['user_id']].append(seq)
    
    print(f"Total users: {len(user_sequences)}")
    
    # Split sequences for each user by timestamp
    train_sequences = []
    val_sequences = []
    
    users_with_val = 0
    users_without_val = 0
    
    for user_id, user_seqs in user_sequences.items():
        # Sort by timestamp
        user_seqs_sorted = sorted(user_seqs, key=lambda x: x['target_timestamp'])
        
        # Calculate split point
        n_total = len(user_seqs_sorted)
        n_train = int(n_total * train_ratio)
        n_val = n_total - n_train
        
        # If validation would have < 1 samples, put all in training
        if n_val < 1:
            train_sequences.extend(user_seqs_sorted)
            users_without_val += 1
        else:
            train_sequences.extend(user_seqs_sorted[:n_train])
            val_sequences.extend(user_seqs_sorted[n_train:])
            users_with_val += 1
    
    # Create output directories
    os.makedirs(Path(train_output_path).parent, exist_ok=True)
    os.makedirs(Path(val_output_path).parent, exist_ok=True)
    
    # Save splits
    write_jsonl(train_output_path, train_sequences, mode='w')
    write_jsonl(val_output_path, val_sequences, mode='w')
    
    print(f"\nSplit summary:")
    print(f"  Training sequences: {len(train_sequences)}")
    print(f"  Validation sequences: {len(val_sequences)}")
    print(f"  Users with validation: {users_with_val}")
    print(f"  Users without validation (all in train): {users_without_val}")
    print(f"\nSaved to:")
    print(f"  Train: {train_output_path}")
    print(f"  Val: {val_output_path}")
    
    return {
        'train': train_sequences,
        'val': val_sequences
    }

@app.command()
def filter_users_by_interaction_count(input_path:str='data/only_ratings_train.csv', output_path:str='data/exp1/special_users.json'):
    """Get the IDs of top 5 users who watch movies the most and the least, requiring users to appear in the test sequences file."""
    import pandas as pd
    import json
    from pathlib import Path
    from utils import read_jsonl
    
    # Read the training dataset
    print(f"Reading training data from: {input_path}")
    df_train = pd.read_csv(input_path)
    
    # Read the test sequences to get users present in test
    test_sequences_path = 'data/exp1/test.jsonl'
    print(f"Reading test sequences from: {test_sequences_path}")
    test_sequences = read_jsonl(test_sequences_path)
    
    # Get unique users from test sequences
    test_users = set(seq['user_id'] for seq in test_sequences)
    print(f"Users in test sequences: {len(test_users)}")
    
    # Count interactions per user in training data
    interaction_counts = df_train.groupby('userId').size().reset_index(name='count')
    print(f"Total users in training: {len(interaction_counts)}")
    
    # Filter to only users present in both train and test sequences
    interaction_counts = interaction_counts[interaction_counts['userId'].isin(test_users)]
    print(f"Users present in both train and test sequences: {len(interaction_counts)}")
    
    if len(interaction_counts) < 5:
        print("Warning: Fewer than 5 users found in both datasets. Adjusting to available users.")
    
    # Get top 5 most active users (or all if fewer)
    num_users = min(5, len(interaction_counts))
    most_active = interaction_counts.sort_values('count', ascending=False).head(num_users)
    most_active_users = most_active['userId'].tolist()
    most_active_counts = most_active['count'].tolist()
    
    # Get top 5 least active users (or all if fewer)
    least_active = interaction_counts.sort_values('count', ascending=True).head(num_users)
    least_active_users = least_active['userId'].tolist()
    least_active_counts = least_active['count'].tolist()
    
    # Prepare data for JSON
    data = {
        'most_active_users': {
            'user_ids': most_active_users,
            'interaction_counts': most_active_counts
        },
        'least_active_users': {
            'user_ids': least_active_users,
            'interaction_counts': least_active_counts
        }
    }
    
    # Create output directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Saved user interaction analysis to: {output_path}")
    print(f"\nMost active users (top {num_users}):")
    for uid, count in zip(most_active_users, most_active_counts):
        print(f"  User {uid}: {count} interactions")
    print(f"\nLeast active users (top {num_users}):")
    for uid, count in zip(least_active_users, least_active_counts):
        print(f"  User {uid}: {count} interactions")

@app.command()
def extract_evaluation_data_for_special_users(special_users_path:str='data/exp1/special_users.json', test_sequences_path:str='data/exp1/test.jsonl', output_path:str='data/exp1/special_users_test_sequences.jsonl'):
    """Extract test sequences for special users identified previously."""
    import json
    from utils import read_jsonl, write_jsonl
    from pathlib import Path
    
    # Load special users
    print(f"Loading special users from: {special_users_path}")
    with open(special_users_path, 'r') as f:
        special_users_data = json.load(f)
    
    special_user_ids = set(special_users_data['most_active_users']['user_ids'] + 
                           special_users_data['least_active_users']['user_ids'])
    print(f"Total special users: {len(special_user_ids)}")
    
    # Load test sequences
    print(f"Loading test sequences from: {test_sequences_path}")
    test_sequences = read_jsonl(test_sequences_path)
    
    # Filter sequences for special users
    filtered_sequences = [seq for seq in test_sequences if seq['user_id'] in special_user_ids]
    print(f"Extracted {len(filtered_sequences)} sequences for special users")
    
    # Create output directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save filtered sequences
    write_jsonl(output_path, filtered_sequences, mode='w')
    print(f"Saved filtered sequences to: {output_path}")
    
    return filtered_sequences

@app.command()
def compute_special_user_metrics(special_users_info_path:str='data/exp1/special_users.json', special_user_result_path:str='output/20260119_230632/per_user_test_metrics.json'):
    """Compute average metrics for special user groups (most active and least active users)."""
    import json
    
    # Load special users info
    print(f"Loading special users from: {special_users_info_path}")
    with open(special_users_info_path, 'r') as f:
        special_users_data = json.load(f)
    
    most_active_user_ids = special_users_data['most_active_users']['user_ids']
    least_active_user_ids = special_users_data['least_active_users']['user_ids']
    
    print(f"Most active users: {most_active_user_ids}")
    print(f"Least active users: {least_active_user_ids}")
    
    # Load per-user test metrics
    print(f"\nLoading per-user metrics from: {special_user_result_path}")
    with open(special_user_result_path, 'r') as f:
        per_user_metrics = json.load(f)
    
    def compute_group_metrics(user_ids, group_name):
        """Compute average metrics for a group of users."""
        total_samples = 0
        metric_sums = {}
        users_found = []
        users_not_found = []
        
        for user_id in user_ids:
            user_key = str(user_id)
            if user_key in per_user_metrics:
                users_found.append(user_id)
                user_data = per_user_metrics[user_key]
                num_samples = user_data['num_samples']
                total_samples += num_samples
                
                # Accumulate weighted metrics (weighted by number of samples)
                for metric_name, metric_value in user_data['avg_metrics'].items():
                    if metric_name not in metric_sums:
                        metric_sums[metric_name] = 0.0
                    metric_sums[metric_name] += metric_value * num_samples
            else:
                users_not_found.append(user_id)
        
        if total_samples == 0:
            print(f"\nWarning: No samples found for {group_name}")
            return None
        
        # Compute weighted averages
        avg_metrics = {metric: value / total_samples for metric, value in metric_sums.items()}
        
        return {
            'group_name': group_name,
            'num_users': len(users_found),
            'total_samples': total_samples,
            'avg_metrics': avg_metrics,
            'users_found': users_found,
            'users_not_found': users_not_found
        }
    
    # Compute metrics for most active users
    print("\n" + "="*60)
    print("Computing metrics for MOST ACTIVE users...")
    print("="*60)
    most_active_results = compute_group_metrics(most_active_user_ids, "most_active_users")
    
    if most_active_results:
        print(f"\nGroup: {most_active_results['group_name']}")
        print(f"Users found: {most_active_results['num_users']}/{len(most_active_user_ids)}")
        print(f"Total test samples: {most_active_results['total_samples']}")
        print(f"\nAverage Metrics:")
        for metric, value in sorted(most_active_results['avg_metrics'].items()):
            print(f"  {metric}: {value:.4f}")
        if most_active_results['users_not_found']:
            print(f"\nUsers not found in results: {most_active_results['users_not_found']}")
    
    # Compute metrics for least active users
    print("\n" + "="*60)
    print("Computing metrics for LEAST ACTIVE users...")
    print("="*60)
    least_active_results = compute_group_metrics(least_active_user_ids, "least_active_users")
    
    if least_active_results:
        print(f"\nGroup: {least_active_results['group_name']}")
        print(f"Users found: {least_active_results['num_users']}/{len(least_active_user_ids)}")
        print(f"Total test samples: {least_active_results['total_samples']}")
        print(f"\nAverage Metrics:")
        for metric, value in sorted(least_active_results['avg_metrics'].items()):
            print(f"  {metric}: {value:.4f}")
        if least_active_results['users_not_found']:
            print(f"\nUsers not found in results: {least_active_results['users_not_found']}")
    
    # Summary comparison
    print("\n" + "="*60)
    print("SUMMARY COMPARISON")
    print("="*60)
    
    if most_active_results and least_active_results:
        print(f"\n{'Metric':<15} {'Most Active':<15} {'Least Active':<15} {'Difference':<15}")
        print("-" * 60)
        for metric in sorted(most_active_results['avg_metrics'].keys()):
            most_val = most_active_results['avg_metrics'][metric]
            least_val = least_active_results['avg_metrics'][metric]
            diff = most_val - least_val
            print(f"{metric:<15} {most_val:<15.4f} {least_val:<15.4f} {diff:+.4f}")
    
    return {
        'most_active_users': most_active_results,
        'least_active_users': least_active_results
    }

@app.command()
def group_by_percentiles(train_data_path:str='data/only_ratings_train.csv', test_data_path:str='data/exp1/test.jsonl', save_path:str='data/exp1/percentile_user_groups.json'):
    """Group users into percentile-based groups based on their interaction counts in the training data. The users id must also appear in the test data."""
    import pandas as pd
    import json
    import numpy as np
    from pathlib import Path
    from utils import read_jsonl
    percentiles:list=[25,50,75,100]
    # Read training data
    print(f"Reading training data from: {train_data_path}")
    df_train = pd.read_csv(train_data_path)
    
    # Read test sequences to get users present in test
    print(f"Reading test sequences from: {test_data_path}")
    test_sequences = read_jsonl(test_data_path)
    test_users = set(seq['user_id'] for seq in test_sequences)
    print(f"Users in test data: {len(test_users)}")
    
    # Count interactions per user in training data
    interaction_counts = df_train.groupby('userId').size().reset_index(name='count')
    print(f"Total users in training: {len(interaction_counts)}")
    
    # Filter to only users present in both train and test
    interaction_counts = interaction_counts[interaction_counts['userId'].isin(test_users)]
    print(f"Users present in both train and test: {len(interaction_counts)}")
    
    # Sort by interaction count
    interaction_counts = interaction_counts.sort_values('count', ascending=True).reset_index(drop=True)
    
    # Create percentile groups
    percentile_groups = {}
    prev_percentile = 0
    
    for percentile in sorted(percentiles):
        # Calculate percentile values
        lower_bound = np.percentile(interaction_counts['count'], prev_percentile)
        upper_bound = np.percentile(interaction_counts['count'], percentile)
        
        # Filter users in this percentile range
        if percentile == 100:
            # Include upper bound for the last group
            group_users = interaction_counts[
                (interaction_counts['count'] >= lower_bound) & 
                (interaction_counts['count'] <= upper_bound)
            ]
        else:
            # Exclude upper bound for intermediate groups
            group_users = interaction_counts[
                (interaction_counts['count'] >= lower_bound) & 
                (interaction_counts['count'] < upper_bound)
            ]
        
        group_name = f"p{prev_percentile}-{percentile}"
        percentile_groups[group_name] = {
            'percentile_range': [prev_percentile, percentile],
            'interaction_count_range': [float(lower_bound), float(upper_bound)],
            'num_users': len(group_users),
            'user_ids': group_users['userId'].tolist(),
            'interaction_counts': group_users['count'].tolist()
        }
        
        print(f"\nGroup {group_name}:")
        print(f"  Interaction range: [{lower_bound:.1f}, {upper_bound:.1f}]")
        print(f"  Number of users: {len(group_users)}")
        
        prev_percentile = percentile
    
    # Create output directory if needed
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save to JSON
    with open(save_path, 'w') as f:
        json.dump(percentile_groups, f, indent=2)
    
    print(f"\nSaved percentile groups to: {save_path}")
    print(f"Total groups created: {len(percentile_groups)}")
    
    return percentile_groups

@app.command()
def plot_user_group_distribution(percentile_groups_path:str='data/exp1/percentile_user_groups.json', save_path:str='output/figures/user_group_distribution.png'):
    """Plot the distribution of users across different percentile-based groups. Using box plot """
    import json
    import matplotlib.pyplot as plt
    from pathlib import Path
    
    # Load percentile groups data
    print(f"Loading percentile groups from: {percentile_groups_path}")
    with open(percentile_groups_path, 'r') as f:
        percentile_groups = json.load(f)
    
    # Prepare data for box plot
    data_to_plot = []
    labels = []
    
    # Sort groups by percentile range
    sorted_groups = sorted(percentile_groups.items(), 
                          key=lambda x: x[1]['percentile_range'][0])
    
    for group_name, group_data in sorted_groups:
        interaction_counts = group_data['interaction_counts']
        # Cap interaction counts at 100 for better visualization
        capped_counts = [min(count, 100) for count in interaction_counts]
        data_to_plot.append(capped_counts)
        labels.append(group_name)
        print(f"Group {group_name}: {len(interaction_counts)} users")
    
    # Create figure and plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create box plot
    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
                    showmeans=True, meanline=False,
                    medianprops=dict(color='red', linewidth=2),
                    meanprops=dict(marker='D', markerfacecolor='green', 
                                  markeredgecolor='green', markersize=6))
    
    # Color the boxes
    colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
        patch.set_facecolor(color)
    
    # Customize plot
    ax.set_xlabel('Percentile Groups', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Interactions', fontsize=12, fontweight='bold')
    ax.set_title('User Interaction Distribution by Percentile Groups', 
                fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', linewidth=2, label='Median'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='green',
               markeredgecolor='green', markersize=6, label='Mean')
    ]
    ax.legend(handles=legend_elements, loc='upper left')
    
    # Adjust layout
    plt.tight_layout()
    
    # Create output directory if needed
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved plot to: {save_path}")
    
    # Also display statistics
    print("\nGroup Statistics:")
    print(f"{'Group':<12} {'Count':<8} {'Min':<8} {'Q1':<8} {'Median':<8} {'Q3':<8} {'Max':<8} {'Mean':<8}")
    print("-" * 80)
    
    import numpy as np
    for label, data in zip(labels, data_to_plot):
        if data:
            stats = {
                'min': np.min(data),
                'q1': np.percentile(data, 25),
                'median': np.median(data),
                'q3': np.percentile(data, 75),
                'max': np.max(data),
                'mean': np.mean(data)
            }
            print(f"{label:<12} {len(data):<8} {stats['min']:<8.1f} {stats['q1']:<8.1f} "
                  f"{stats['median']:<8.1f} {stats['q3']:<8.1f} {stats['max']:<8.1f} {stats['mean']:<8.1f}")
    
    plt.close()
    return fig

@app.command()
def compute_user_group_metrics(eval_results_path:str='output/20260126_220135/per_user_test_metrics.json', special_user_groups_path:str='data/exp1/special_users.json'):
    """Compute average metrics for user groups defined in the special_user_groups_path JSON file."""
    import json
    
    # Load user groups
    print(f"Loading user groups from: {special_user_groups_path}")
    with open(special_user_groups_path, 'r') as f:
        user_groups_data = json.load(f)
    
    # Load per-user test metrics
    print(f"Loading per-user metrics from: {eval_results_path}")
    with open(eval_results_path, 'r') as f:
        per_user_metrics = json.load(f)
    
    def compute_group_metrics(user_ids, group_name):
        """Compute average metrics for a group of users."""
        total_samples = 0
        metric_sums = {}
        users_found = []
        users_not_found = []
        
        for user_id in user_ids:
            user_key = str(user_id)
            if user_key in per_user_metrics:
                users_found.append(user_id)
                user_data = per_user_metrics[user_key]
                num_samples = user_data['num_samples']
                total_samples += num_samples
                
                # Accumulate weighted metrics (weighted by number of samples)
                for metric_name, metric_value in user_data['avg_metrics'].items():
                    if metric_name not in metric_sums:
                        metric_sums[metric_name] = 0.0
                    metric_sums[metric_name] += metric_value * num_samples
            else:
                users_not_found.append(user_id)
        
        if total_samples == 0:
            print(f"Warning: No samples found for {group_name}")
            return None
        
        # Compute weighted averages
        avg_metrics = {metric: value / total_samples for metric, value in metric_sums.items()}
        
        return {
            'group_name': group_name,
            'num_users': len(users_found),
            'total_samples': total_samples,
            'avg_metrics': avg_metrics,
            'users_found': users_found,
            'users_not_found': users_not_found
        }
    
    # Compute metrics for each group
    group_results = {}
    for group_name, group_data in user_groups_data.items():
        print(f"\n{'='*60}")
        print(f"Computing metrics for group: {group_name}")
        print(f"{'='*60}")
        
        user_ids = group_data['user_ids']
        results = compute_group_metrics(user_ids, group_name)
        
        if results:
            print(f"Group: {results['group_name']}")
            print(f"Users found: {results['num_users']}/{len(user_ids)}")
            print(f"Total test samples: {results['total_samples']}")
            print("Average Metrics:" )
            for metric, value in sorted(results['avg_metrics'].items()):
                print(f"  {metric}: {value:.4f}")
            if results['users_not_found']:
                print(f"Users not found in results: {results['users_not_found']}")
        
        group_results[group_name] = results
    
    # Summary comparison
    print(f"\n{'='*60}")
    print("SUMMARY COMPARISON")
    print(f"{'='*60}")
    
    # Get all metric names
    all_metrics = set()
    for result in group_results.values():
        if result:
            all_metrics.update(result['avg_metrics'].keys())
    
    if all_metrics:
        print(f"\n{'Metric':<15} {' | '.join(f'{group:<12}' for group in sorted(group_results.keys()))}")
        print("-" * (15 + 3 + sum(12 for _ in group_results)))
        
        for metric in sorted(all_metrics):
            values = []
            for group_name in sorted(group_results.keys()):
                result = group_results[group_name]
                if result and metric in result['avg_metrics']:
                    values.append(f"{result['avg_metrics'][metric]:.4f}")
                else:
                    values.append("N/A")
            print(f"{metric:<15} {' | '.join(f'{v:<12}' for v in values)}")
    
    return group_results


if __name__ == "__main__":
    app()