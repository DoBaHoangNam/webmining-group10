# Training Code Updates

## Summary of Changes

The training code has been modified with three major improvements:

### 1. AdamW Optimizer
- **Changed from:** `torch.optim.Adam`
- **Changed to:** `torch.optim.AdamW`
- **Why:** AdamW provides better weight decay implementation, leading to improved generalization and model performance.

### 2. Configurable Metrics Class
- **New class:** `RecMetrics` 
- **Purpose:** Separate metrics computation into a reusable class with configurable top-k values
- **Metrics computed:**
  - Accuracy (top-1 prediction)
  - Hit@k (whether positive item is in top-k)
  - NDCG@k (Normalized Discounted Cumulative Gain)
  - MRR@k (Mean Reciprocal Rank)

### 3. Best Model Tracking
- **New callback:** `on_validation_epoch_end()`
- **Features:**
  - Evaluates model after each epoch
  - Tracks best validation accuracy
  - Automatically saves model when validation accuracy improves
  - Creates checkpoint files named: `Mamba4Rec-best-acc-{accuracy}.pth`

## Configuration

### Setting Top-K Values

In your `config.yaml`, you can configure which top-k metrics to compute:

```yaml
topk: [5, 10, 20]  # Will compute Hit@5, Hit@10, Hit@20, etc.
```

The default is `[10]` if not specified.

### Model Checkpointing

Two types of checkpoints are now saved:

1. **Best accuracy checkpoint:**
   - Filename: `Mamba4Rec-{epoch:02d}-acc-{val_accuracy:.4f}.pth`
   - Monitors: `val_accuracy` (max)
   - Saves: Top 1 model

2. **Best loss checkpoint:**
   - Filename: `Mamba4Rec-{epoch:02d}-loss-{val_loss:.4f}.pth`
   - Monitors: `val_loss` (min)
   - Saves: Top 1 model + last checkpoint

3. **Best accuracy (manual save):**
   - Filename: `Mamba4Rec-best-acc-{val_accuracy:.4f}.pth`
   - Saved in `on_validation_epoch_end()` callback
   - Contains only model weights

### Early Stopping

Early stopping is now enabled and monitors validation accuracy:

```yaml
stopping_step: 10  # Stops if no improvement after 10 epochs
```

## Usage Example

```bash
python train.py \
  --config config.yaml \
  --train_data data/exp1/train_split.jsonl \
  --val_data data/exp1/val_split.jsonl
```

## Metrics Logged to TensorBoard

- `train_loss` (per step and per epoch)
- `val_loss` (per epoch)
- `val_accuracy` (per epoch)
- `val_hit@k` for each k in topk (per epoch)
- `val_ndcg@k` for each k in topk (per epoch)
- `val_mrr@k` for each k in topk (per epoch)

## Code Structure

```python
# RecMetrics class handles all metric computations
class RecMetrics:
    def __init__(self, topk: List[int] = [5, 10, 20])
    def compute_metrics(self, logits, positive_idx=0) -> Dict[str, float]

# Lightning module uses RecMetrics
class Mamba4RecLightning(pl.LightningModule):
    def __init__(self, config):
        self.metrics_calculator = RecMetrics(topk=config.get("topk", [10]))
        self.best_val_accuracy = 0.0
    
    def validation_step(self, batch, batch_idx):
        # Compute metrics using metrics_calculator
        metrics = self.metrics_calculator.compute_metrics(logits)
        
    def on_validation_epoch_end(self):
        # Save best model if validation accuracy improved
        
    def configure_optimizers(self):
        # Use AdamW optimizer
        optimizer = torch.optim.AdamW(...)
```

## Benefits

1. **Better optimization:** AdamW provides improved weight decay handling
2. **Flexible evaluation:** Easy to add/remove metrics or change top-k values
3. **Best model tracking:** Automatically saves best performing models
4. **Rich metrics:** Multiple metrics (Hit, NDCG, MRR) at different k values
5. **Early stopping:** Prevents overfitting by stopping when validation stops improving
