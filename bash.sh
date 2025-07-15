deepspeed --num_gpus 2 train_coles_universal.py

torchrun --nproc_per_node=1 finetune_coles_universal.py