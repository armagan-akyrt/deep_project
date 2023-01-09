python3 train.py \
    --model_save_path "model"\
    --dataset_path "data"\
    --num_train_epoch "1"


python3 pipeline.py \
    --model_load_path "model"\
    --input_file "input.txt" \
    --output_file "output.txt"\
    --checkpoint "12500"
