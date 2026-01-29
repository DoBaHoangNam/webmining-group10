class RecommendationSchema:
    user_id: int
    target_item_id: int
    interaction_history: list[int]
    negative_samples: list[int]
    history_length: int
    num_negatives: int