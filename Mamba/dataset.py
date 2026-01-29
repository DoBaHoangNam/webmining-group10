from torch.utils.data import Dataset
from utils import read_jsonl
class RecDataset(Dataset):
    def __init__(self, data_path:str, max_history_length:int=50):
        """
        Initializes the dataset with user-item interactions.

        Args:
            user_item_interactions (list of tuples): Each tuple contains (user_id, item_id).
        """
        self.data = read_jsonl(data_path)
        max_history_length = 0
        max_negatives = 0
        for record in self.data:
            record['interaction_history'] = record['interaction_history'][-max_history_length:]
            record['history_length'] = len(record.get('interaction_history', []))
            record['num_negatives'] = len(record.get('negative_samples', []))
            max_history_length = max(max_history_length, record['history_length'])
            max_negatives = max(max_negatives, record['num_negatives'])
        
        # pad interaction_history and negative_samples to max lengths
        for record in self.data:
            history = record.get('interaction_history', [])
            negatives = record.get('negative_samples', [])
            record['interaction_history'] = history + [0] * (max_history_length - len(history))
            record['negative_samples'] = negatives + [0] * (max_negatives - len(negatives))
            

    def __len__(self):
        """
        Returns the total number of interactions in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves the user-item interaction at the specified index.

        Args:
            idx (int): Index of the interaction to retrieve.

        """
        return {
            "user_id": self.data[idx]['user_id'],
            "interaction_history": self.data[idx]['interaction_history'],
            "target_item_id": self.data[idx]['target_item_id'],
            "negative_samples": self.data[idx]['negative_samples'],
            "history_length": self.data[idx]['history_length'],
            "num_negatives": self.data[idx]['num_negatives']
        }

