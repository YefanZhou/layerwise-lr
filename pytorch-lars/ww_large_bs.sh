

export PYTHONPATH="${PYTHONPATH}:/home/eecs/yefan0726/layer-wise-learning-rate-schedule-/ww_tianyu"

for bs in 128 256 1024 2048 4096
    do
        for ep in 0 10 20 30 40 50 60 70 80 90 100
            do
                echo $bs $ep
                OMP_NUM_THREADS=1 python model_layer_evaluate_ww_large_bs.py \
                    --resume /data/yefan0726/checkpoints/large_batch/_optim_sgd_CIFAR100_${bs}_0.1/model_ep${ep}.ckpt \
                    --model 'resnet18' \
                    --seed 42 \
                    --downsample \
                    --save_dir /data/yefan0726/checkpoints/large_batch/_optim_sgd_CIFAR100_${bs}_0.1/layer_stats/ww/model_ep${ep} &
            done
            wait
    done
