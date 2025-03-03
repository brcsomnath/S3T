device="${1:-1}"
start="${2:-0}"
num_shards="${3:-4}"
loras="${4:-8}"
rank="${5:-32}"
model="${6:-llama-13b}"
for (( i=$start ; i<$num_shards ; i++ )); 
do
    CUDA_VISIBLE_DEVICES=$device python ../src/s3t_llm.py \
     --shard_id $i \
     --num_loras $loras \
     --rank $rank \
     --num_shards $num_shards \
     --model $model
done
