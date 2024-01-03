model_name_or_path="multilingual-e5-large"
train_batch_size=1
num_epochs=5
model_save_path="output/sbert-v1.0"
train_file=/data/sample.json"

python train_sbert.py \
    --model_name_or_path ${model_name_or_path} \
    --train_batch_size ${train_batch_size} \
    --num_epochs ${num_epochs} \
    --model_save_path ${model_save_path} \
    --train_file ${train_file}
