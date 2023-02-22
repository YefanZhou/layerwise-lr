


count=0
for bs in 128 256 1024 2048 4096
    do
        for ep in 0 10 20 30 40 50 60 70 80 90 100
            do
                id=$((count%8))
                echo $bs $ep $id
                CUDA_VISIBLE_DEVICES=$id python model_layer_evaluate_svm_large_bs.py \
                    --resume /data/yefan0726/checkpoints/large_batch/_optim_sgd_CIFAR100_${bs}_0.1/model_ep${ep}.ckpt \
                    --model 'resnet18' \
                    --seed 42 \
                    --downsample \
                    --save_dir /data/yefan0726/checkpoints/large_batch/_optim_sgd_CIFAR100_${bs}_0.1/layer_stats/svm/model_ep${ep} &
                let count++
            done
            wait
    done
