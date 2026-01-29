import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import yaml
import argparse
from pathlib import Path
from typing import List, Dict
import numpy as np
from datetime import datetime

from model import Mamba4Rec
from dataset import RecDataset


def collate_fn(batch):
    """
    Custom collate function to batch data.
    
    Args:
        batch: List of dictionaries from dataset __getitem__
        
    Returns:
        Dictionary with batched tensors
    """
    user_ids = torch.tensor([item['user_id'] for item in batch], dtype=torch.long)
    interaction_history = torch.tensor([item['interaction_history'] for item in batch], dtype=torch.long)
    target_item_id = torch.tensor([item['target_item_id'] for item in batch], dtype=torch.long)
    negative_samples = torch.tensor([item['negative_samples'] for item in batch], dtype=torch.long)
    history_length = torch.tensor([item['history_length'] for item in batch], dtype=torch.long)
    num_negatives = torch.tensor([item['num_negatives'] for item in batch], dtype=torch.long)
    
    return {
        'user_id': user_ids,
        'interaction_history': interaction_history,
        'target_item_id': target_item_id,
        'negative_samples': negative_samples,
        'history_length': history_length,
        'num_negatives': num_negatives
    }


class RecMetrics:
    """Class for computing recommendation metrics with configurable topk."""
    
    def __init__(self, topk: List[int] = [5, 10, 20]):
        """
        Initialize metrics calculator.
        
        Args:
            topk: List of k values for computing metrics (e.g., [5, 10, 20])
        """
        self.topk = topk
    
    def compute_metrics(self, logits: torch.Tensor, positive_idx: int = 0) -> Dict[str, float]:
        """
        Compute ranking metrics for a batch of predictions.
        
        Args:
            logits: Tensor of shape [B, N] containing scores for items
            positive_idx: Index of positive item (default: 0)
        
        Returns:
            Dictionary containing computed metrics
        """
        metrics = {}
        batch_size = logits.size(0)
        
        # Compute accuracy (whether positive item ranks first)
        predictions = torch.argmax(logits, dim=1)
        accuracy = (predictions == positive_idx).float().mean()
        metrics['accuracy'] = accuracy.item()
        
        # Compute metrics for each k
        for k in self.topk:
            k_actual = min(k, logits.size(1))
            _, top_k_indices = torch.topk(logits, k=k_actual, dim=1)  # [B, k]
            
            # Hit@k: Whether positive item is in top-k
            hit_at_k = (top_k_indices == positive_idx).any(dim=1).float().mean()
            metrics[f'hit@{k}'] = hit_at_k.item()
            
            # NDCG@k: Normalized Discounted Cumulative Gain
            positions = (top_k_indices == positive_idx).nonzero(as_tuple=True)
            ndcg_scores = torch.zeros(batch_size, device=logits.device)
            if len(positions[0]) > 0:
                for i, pos in zip(positions[0], positions[1]):
                    ndcg_scores[i] = 1.0 / torch.log2(pos.float() + 2)  # +2 because position is 0-indexed
            ndcg_at_k = ndcg_scores.mean()
            metrics[f'ndcg@{k}'] = ndcg_at_k.item()
            
            # MRR@k: Mean Reciprocal Rank
            mrr_scores = torch.zeros(batch_size, device=logits.device)
            if len(positions[0]) > 0:
                for i, pos in zip(positions[0], positions[1]):
                    mrr_scores[i] = 1.0 / (pos.float() + 1)  # +1 because position is 0-indexed
            mrr_at_k = mrr_scores.mean()
            metrics[f'mrr@{k}'] = mrr_at_k.item()
        
        return metrics


class Mamba4RecLightning(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        
        # Initialize model
        self.model = Mamba4Rec(config)
        
        # Training configuration
        self.learning_rate = config["learning_rate"]
        self.weight_decay = config.get("weight_decay", 0.0)
        
        # Initialize metrics calculator
        topk = config.get("topk", [10])
        self.metrics_calculator = RecMetrics(topk=topk)
        
        # Track best validation MRR@5
        self.best_val_mrr5 = 0.0
        
        # Store test outputs for on_test_epoch_end
        self.test_outputs = []
        
    def forward(self, item_seq, item_seq_len, user_ids):
        return self.model(item_seq, item_seq_len, user_ids)
    
    def training_step(self, batch, batch_idx):
        # Extract data from batch
        user_ids = batch['user_id'].to(self.device)
        interaction_history = batch['interaction_history'].to(self.device)
        target_item_id = batch['target_item_id'].to(self.device)
        negative_samples = batch['negative_samples'].to(self.device)
        item_seq_len = batch['history_length'].to(self.device)
        # Forward pass
        seq_output = self.forward(interaction_history, item_seq_len, user_ids)
        
        # Concatenate positive and negative samples
        # Positive item is placed at index 0
        item_ids = torch.cat([target_item_id.unsqueeze(1), negative_samples], dim=1)  # [B, N+1]
        
        # Compute loss
        loss = self.model.compute_loss(seq_output, item_ids)
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        # Extract data from batch
        user_ids = batch['user_id'].to(self.device)
        interaction_history = batch['interaction_history'].to(self.device)
        target_item_id = batch['target_item_id'].to(self.device)
        negative_samples = batch['negative_samples'].to(self.device)
        item_seq_len = batch['history_length'].to(self.device)
        
        # Forward pass
        seq_output = self.forward(interaction_history, item_seq_len, user_ids)
        
        # Concatenate positive and negative samples
        item_ids = torch.cat([target_item_id.unsqueeze(1), negative_samples], dim=1)  # [B, N+1]
        
        # Compute loss
        loss, logits = self.model.compute_loss(seq_output, item_ids, return_logits=True)
        
        # Compute all metrics using metrics calculator
        metrics = self.metrics_calculator.compute_metrics(logits, positive_idx=0)
        
        # Log metrics
        self.log('val_loss', loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log('val_accuracy', metrics['accuracy'], prog_bar=True, on_epoch=True, sync_dist=True)
        
        # Log all topk metrics
        for k in self.metrics_calculator.topk:
            self.log(f'val_hit@{k}', metrics[f'hit@{k}'], on_epoch=True, sync_dist=True)
            self.log(f'val_ndcg@{k}', metrics[f'ndcg@{k}'], prog_bar=True, on_epoch=True, sync_dist=True)
            self.log(f'val_mrr@{k}', metrics[f'mrr@{k}'], on_epoch=True, sync_dist=True)
        
        return {'val_loss': loss, 'val_accuracy': metrics['accuracy']}
    
    def on_validation_epoch_end(self):
        """Called at the end of validation epoch."""
        # Track best MRR@5 for logging purposes
        current_val_mrr5 = self.trainer.callback_metrics.get('val_mrr@5', 0.0)
        
        if isinstance(current_val_mrr5, torch.Tensor):
            current_val_mrr5 = current_val_mrr5.item()
        
        if current_val_mrr5 > self.best_val_mrr5:
            self.best_val_mrr5 = current_val_mrr5
            print(f"\nNew best MRR@5: {self.best_val_mrr5:.4f}")
    
    def test_step(self, batch, batch_idx):
        """Test step for final evaluation."""
        # Extract data from batch
        user_ids = batch['user_id'].to(self.device)
        interaction_history = batch['interaction_history'].to(self.device)
        target_item_id = batch['target_item_id'].to(self.device)
        negative_samples = batch['negative_samples'].to(self.device)
        item_seq_len = batch['history_length'].to(self.device)
        
        # Forward pass
        seq_output = self.forward(interaction_history, item_seq_len, user_ids)
        
        # Concatenate positive and negative samples
        item_ids = torch.cat([target_item_id.unsqueeze(1), negative_samples], dim=1)  # [B, N+1]
        
        # Compute loss
        loss, logits = self.model.compute_loss(seq_output, item_ids, return_logits=True)
        
        # Compute all metrics using metrics calculator
        metrics = self.metrics_calculator.compute_metrics(logits, positive_idx=0)
        
        # Log metrics
        self.log('test_loss', loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log('test_accuracy', metrics['accuracy'], prog_bar=True, on_epoch=True, sync_dist=True)
        
        # Log all topk metrics
        for k in self.metrics_calculator.topk:
            self.log(f'test_hit@{k}', metrics[f'hit@{k}'], on_epoch=True, sync_dist=True)
            self.log(f'test_ndcg@{k}', metrics[f'ndcg@{k}'], prog_bar=True, on_epoch=True, sync_dist=True)
            self.log(f'test_mrr@{k}', metrics[f'mrr@{k}'], prog_bar=True, on_epoch=True, sync_dist=True)
        
        # Return user_ids and individual metrics for per-user aggregation
        batch_size = user_ids.size(0)
        individual_metrics = []
        for i in range(batch_size):
            user_metrics = {'user_id': user_ids[i].item()}
            # Compute individual metrics for this sample
            single_logits = logits[i:i+1]  # [1, N+1]
            single_metrics = self.metrics_calculator.compute_metrics(single_logits, positive_idx=0)
            user_metrics.update(single_metrics)
            individual_metrics.append(user_metrics)
        
        output = {
            'test_loss': loss, 
            'test_accuracy': metrics['accuracy'],
            'individual_metrics': individual_metrics
        }
        
        # Store output for on_test_epoch_end
        self.test_outputs.append(output)
        
        return output
    
    def on_test_epoch_end(self):
        """Aggregate metrics per user at the end of test epoch."""
        from collections import defaultdict
        
        # Access outputs from instance attribute
        outputs = self.test_outputs
        
        # Collect all individual metrics
        all_individual_metrics = []
        for output in outputs:
            all_individual_metrics.extend(output['individual_metrics'])
        
        # Group metrics by user_id
        user_metrics = defaultdict(list)
        for metrics in all_individual_metrics:
            user_id = metrics['user_id']
            user_metrics[user_id].append(metrics)
        
        # Compute average metrics per user
        per_user_results = {}
        for user_id, metrics_list in user_metrics.items():
            user_avg_metrics = {}
            num_samples = len(metrics_list)
            
            # Average each metric across all samples for this user
            for key in metrics_list[0].keys():
                if key == 'user_id':
                    continue
                values = [m[key] for m in metrics_list]
                user_avg_metrics[key] = sum(values) / len(values)
            
            per_user_results[user_id] = {
                'num_samples': num_samples,
                'avg_metrics': user_avg_metrics
            }
        
        # Save per-user results to file
        import json
        output_file = self.work_dir_path / "per_user_test_metrics.json"
        with open(output_file, 'w') as f:
            json.dump(per_user_results, f, indent=2)
        
        print(f"\nPer-user test metrics saved to: {output_file}")
        print(f"Total users evaluated: {len(per_user_results)}")
        
        # Log summary statistics
        all_hit5 = [user_data['avg_metrics']['hit@5'] for user_data in per_user_results.values()]
        all_ndcg5 = [user_data['avg_metrics']['ndcg@5'] for user_data in per_user_results.values()]
        all_mrr5 = [user_data['avg_metrics']['mrr@5'] for user_data in per_user_results.values()]
        
        print(f"Average Hit@5 per user: {sum(all_hit5)/len(all_hit5):.4f}")
        print(f"Average NDCG@5 per user: {sum(all_ndcg5)/len(all_ndcg5):.4f}")
        print(f"Average MRR@5 per user: {sum(all_mrr5)/len(all_mrr5):.4f}")
        
        # Clear outputs for next test run
        self.test_outputs = []
    
    def configure_optimizers(self):
        # Use AdamW optimizer (Adam with weight decay fix)
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        return {
            'optimizer': optimizer,
            # 'lr_scheduler': {
            #     'scheduler': scheduler,
            #     'monitor': 'val_loss',
            #     'interval': 'epoch',
            #     'frequency': 1
            # }
        }


def train(config: dict, train_data_path: str, val_data_path: str, test_data_path: str = None, 
          checkpoint_path: str = None, eval_only: bool = False, work_dir: str = 'experiments'):
    """
    Main training function.
    
    Args:
        config: Configuration dictionary
        train_data_path: Path to training data JSONL file
        val_data_path: Path to validation data JSONL file (optional)
        test_data_path: Path to test data JSONL file (optional)
        checkpoint_path: Path to checkpoint file to load (optional)
        eval_only: If True, skip training and only evaluate on test set
        work_dir: Working directory for saving results
    """
    
    # Create working directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    work_dir_path = Path(work_dir)
    if eval_only:
        work_dir_path.mkdir(parents=True, exist_ok=True)
    else:
        work_dir_path = work_dir_path / timestamp
        work_dir_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nWorking directory: {work_dir_path}")
    
    # Create datasets and dataloaders only when needed
    train_loader = None
    val_loader = None
    if not eval_only:
        # Create datasets
        train_dataset = RecDataset(train_data_path, config['max_history_length'])
        val_dataset = RecDataset(val_data_path, config['max_history_length'])
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['train_batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['eval_batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn
        )
    
    # Create test dataloader if test data is provided
    test_loader = None
    if test_data_path:
        test_dataset = RecDataset(test_data_path, config['max_history_length'])
        test_loader = DataLoader(
            test_dataset,
            batch_size=config['eval_batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn
        )
    
    # Initialize model
    model = Mamba4RecLightning(config)
    model.work_dir_path = work_dir_path  # Store working directory for saving results
    
    # Setup callbacks
    if not eval_only:
        # Monitor MRR@5 for best model
        checkpoint_callback_best = ModelCheckpoint(
            dirpath=str(work_dir_path),
            filename='best-mrr5-{epoch:02d}-{val_mrr@5:.4f}',
            save_top_k=1,
            monitor='val_mrr@5',
            mode='max',
            verbose=True,
            save_last=False
        )
        
        # Save last epoch checkpoint
        checkpoint_callback_last = ModelCheckpoint(
            dirpath=str(work_dir_path),
            filename='last-epoch-{epoch:02d}',
            save_top_k=1,
            save_last=True,
            verbose=True
        )
        
        callbacks = [checkpoint_callback_best, checkpoint_callback_last]
    else:
        callbacks = []
    
    # Setup logger
    logger = TensorBoardLogger(
        save_dir='log_tensorboard/',
        name=f'Mamba4Rec-{timestamp}'
    )
    
    # Initialize trainer
    if eval_only:
        # Simplified trainer for evaluation only
        trainer = pl.Trainer(
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=[int(config['gpu_id'])] if torch.cuda.is_available() and config.get('gpu_id') else 'auto',
            callbacks=[],
            logger=False,  # No logging for eval_only
            precision='16-mixed' if torch.cuda.is_available() else 32,
        )
    else:
        trainer = pl.Trainer(
            max_epochs=config['epochs'],
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=[int(config['gpu_id'])] if torch.cuda.is_available() and config.get('gpu_id') else 'auto',
            callbacks=callbacks,
            logger=logger,
            log_every_n_steps=10,
            check_val_every_n_epoch=config.get('eval_step', 1),  # Evaluate after each epoch
            gradient_clip_val=1.0,
            precision='16-mixed' if torch.cuda.is_available() else 32,
        )
    
    # Load checkpoint if provided
    if checkpoint_path:
        print(f"\nLoading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        print("Checkpoint loaded successfully")
    
    # Eval-only mode: skip training and evaluate on test set
    if eval_only:
        if not test_loader:
            raise ValueError("Test data path must be provided for eval-only mode")
        print("\nRunning evaluation only (no training)...")
        results = trainer.test(model, test_loader)
        print("\nTest Results:")
        for key, value in results[0].items():
            print(f"  {key}: {value:.4f}")
        
        # Save test results to file
        import json
        test_results_file = work_dir_path / 'test_results.json'
        with open(test_results_file, 'w') as f:
            json.dump(results[0], f, indent=2)
        print(f"Test results saved to: {test_results_file}")
        
        return model, trainer
    
    # Train model
    trainer.fit(model, train_loader, val_loader)
    
    # Evaluate on test set after training if test data is provided
    if test_loader:
        print("\nEvaluating on test set...")
        results = trainer.test(model, test_loader)
        print("\nTest Results:")
        for key, value in results[0].items():
            print(f"  {key}: {value:.4f}")
        
        # Save test results to file
        import json
        test_results_file = work_dir_path / 'test_results.json'
        with open(test_results_file, 'w') as f:
            json.dump(results[0], f, indent=2)
        print(f"Test results saved to: {test_results_file}")
    
    return model, trainer


def parse_extra_args(extra_args):
    """
    Parse extra arguments in the format key1.key2=val1 and update config.
    
    Args:
        extra_args: List of strings in format 'key1.key2=value'
    
    Returns:
        Dictionary with nested structure based on dot-separated keys
    """
    config_updates = {}
    
    if not extra_args:
        return config_updates
    
    for arg in extra_args:
        if '=' not in arg:
            print(f"Warning: Skipping invalid argument format: {arg}")
            continue
        
        key_path, value = arg.split('=', 1)
        keys = key_path.split('.')
        
        # Try to convert value to appropriate type
        try:
            # Try int
            value = int(value)
        except ValueError:
            try:
                # Try float
                value = float(value)
            except ValueError:
                # Keep as string, handle boolean strings
                if value.lower() == 'true':
                    value = True
                elif value.lower() == 'false':
                    value = False
        
        # Build nested dictionary
        current = config_updates
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value
    
    return config_updates


def update_nested_dict(base_dict, updates):
    """
    Update a nested dictionary with values from another nested dictionary.
    
    Args:
        base_dict: Base dictionary to update
        updates: Dictionary with updates to apply
    """
    for key, value in updates.items():
        if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
            update_nested_dict(base_dict[key], value)
        else:
            base_dict[key] = value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Mamba4Rec model with PyTorch Lightning')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--train_data', type=str, default='data/exp1/train.jsonl', help='Path to training data')
    parser.add_argument('--val_data', type=str, default=None, help='Path to validation data')
    parser.add_argument('--test_data', type=str, default=None, help='Path to test data')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint file to load')
    parser.add_argument('--eval_only', action='store_true', help='Only evaluate on test set, no training')
    parser.add_argument('--work_dir', type=str, default='output', help='Working directory for saving results')
    parser.add_argument('--extra_args', nargs='*', help='Extra config overrides in format key1.key2=value')
    args = parser.parse_args()
    
    # Load base config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Apply extra argument overrides
    if args.extra_args:
        config_updates = parse_extra_args(args.extra_args)
        update_nested_dict(config, config_updates)
        print("\nApplied config overrides:")
        print(yaml.dump(config_updates, default_flow_style=False))
    
    # Run training or evaluation
    train(config, args.train_data, args.val_data, args.test_data, args.checkpoint, args.eval_only, args.work_dir)
