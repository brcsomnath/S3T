device="${1:-0}"
start="${2:-0}"
num_shards="${3:-6}"
model="${4:-vit-base}"
dataset="${5:-cifar10}"

for (( i=$start ; i<$num_shards ; i++ )); 
do
    CUDA_VISIBLE_DEVICES=$device python ../src/s3t_vit.py\
     --shard_id $i \
     --num_shards $num_shards \
     --model_checkpoint $model \
     --dataset $dataset \
     --batch_size 64 
done
