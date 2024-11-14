#!/bin/bash

for model_id in {1..10}; do
    case $model_id in
        1) num_layers=2; num_heads=2; embedding_size=64; ff_dim=128 ;;
        2) num_layers=2; num_heads=4; embedding_size=128; ff_dim=256 ;;
        3) num_layers=4; num_heads=4; embedding_size=256; ff_dim=512 ;;
        4) num_layers=4; num_heads=8; embedding_size=512; ff_dim=1024 ;;
        5) num_layers=6; num_heads=8; embedding_size=512; ff_dim=1024 ;;
        6) num_layers=6; num_heads=8; embedding_size=768; ff_dim=1536 ;;
        7) num_layers=8; num_heads=8; embedding_size=768; ff_dim=1536 ;;
        8) num_layers=8; num_heads=12; embedding_size=768; ff_dim=1536 ;;
        9) num_layers=10; num_heads=12; embedding_size=768; ff_dim=1536 ;;
        10) num_layers=12; num_heads=12; embedding_size=768; ff_dim=1536 ;;
    esac
    
    nohup python EmergenceTrainer.py --model_id $model_id --num_layers $num_layers --num_heads $num_heads --embedding_size $embedding_size --ff_dim $ff_dim &
done