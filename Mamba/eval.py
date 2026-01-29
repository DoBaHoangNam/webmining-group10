
# sampled_negatives: list of negative items 
for user in test_users:
    history = train_interactions[user]
    for positive_item, sampled_negatives in test_data[user]:
        gt_item = test_item[user]

        candidates = [gt_item] + sampled_negatives
        scores = model(user, candidates, history)

        ranked_list = sort_by_score(scores)

        compute HR@K, MRR, NDCG@K
average metrics over users