CUDA_VISIBLE_DEVICES=0  python run.py

python tasks.py extract-user-data

python tasks.py reindex-user-item --data-path data/ratings_train_clean.csv --save-path data/only_ratings_train.csv --mapping-path data/index_mappings.json

python tasks.py preprocess-data --input-path data/ratings_test_clean.csv --output-path data/only_ratings_test.csv --index-mapping-path data/index_mappings.json

python tasks.py extract-interaction-history-for-training --input-file data/only_ratings_train.csv --output-file data/exp1/train.jsonl \
    --max-history-length 100 --min-positive-threshold 3.5 --do-augment-interactions --augmentation-type random_positive --num-negative-samples 50
python tasks.py extract-interaction-history-for-evaluation --input-file data/only_ratings_test.csv --train-history-file data/only_ratings_train.csv \
    --output-file data/exp1/test.jsonl --max-history-length 100 --min-positive-threshold 3.5 --num-negative-samples 50

python tasks.py split-train-val --input-path data/exp1/train.jsonl --train-output-path data/exp1/train_split.jsonl --val-output-path data/exp1/val_split.jsonl --train-ratio 0.9

python train.py --config config.yaml --train_data data/exp1/train_split.jsonl --val_data data/exp1/val_split.jsonl --test_data data/exp1/test.jsonl \
    --checkpoint output/20260119_230632/best-mrr5-epoch=395-val_mrr@5=0.5191.ckpt --eval_only --extra_args max_history_length=25 ;\
python train.py --config config.yaml --train_data data/exp1/train_split.jsonl --val_data data/exp1/val_split.jsonl --test_data data/exp1/test.jsonl \
    --checkpoint output/20260119_231701/best-mrr5-epoch=394-val_mrr@5=0.4958.ckpt --eval_only --extra_args max_history_length=50 ;\
python train.py --config config.yaml --train_data data/exp1/train_split.jsonl --val_data data/exp1/val_split.jsonl --test_data data/exp1/test.jsonl \
    --checkpoint output/20260119_232818/best-mrr5-epoch=396-val_mrr@5=0.5022.ckpt --eval_only --extra_args max_history_length=75



python train.py --config config.yaml --train_data data/exp1/train_split.jsonl --val_data data/exp1/val_split.jsonl --test_data data/exp1/test.jsonl \
    --extra_args max_history_length=25 ;\
python train.py --config config.yaml --train_data data/exp1/train_split.jsonl --val_data data/exp1/val_split.jsonl --test_data data/exp1/test.jsonl \
    --extra_args max_history_length=50 ;\
python train.py --config config.yaml --train_data data/exp1/train_split.jsonl --val_data data/exp1/val_split.jsonl --test_data data/exp1/test.jsonl \
    --extra_args max_history_length=75 

# experiment with different number of layers
python train.py --config config.yaml --train_data data/exp1/train_split.jsonl --val_data data/exp1/val_split.jsonl --test_data data/exp1/test.jsonl \
    --extra_args max_history_length=25 num_layers=2; \

python train.py --config config.yaml --train_data data/exp1/train_split.jsonl --val_data data/exp1/val_split.jsonl --test_data data/exp1/test.jsonl \
    --extra_args max_history_length=100 num_layers=2
# experiment with large hidden size


tensorboard --logdir ./log_tensorboard

python tasks.py filter-users-by-interaction-count

# extract all interaction history for sepecial user
python tasks.py extract-evaluation-data-for-special-users 
# Evaluate and compute hr@5, mrr@5 
python train.py --eval_only --work_dir output/20260119_230632 --test_data data/exp1/test.jsonl --checkpoint output/20260119_230632/best-mrr5-epoch=395-val_mrr@5=0.5191.ckpt
python train.py --eval_only --work_dir output/20260119_230632 --test_data data/exp1/special_users_test_sequences.jsonl --checkpoint output/20260119_230632/best-mrr5-epoch=395-val_mrr@5=0.5191.ckpt

# Làm nốt slide phần mamba

# Làm method phần mamba

python tasks.py compute-user-group-metrics --eval-results-path output/20260127_223958/per_user_test_metrics.json --special-user-groups-path data/exp1/special_users.json
python tasks.py compute-user-group-metrics --eval-results-path output/20260127_223958/per_user_test_metrics.json --special-user-groups-path data/exp1/percentile_user_groups.json