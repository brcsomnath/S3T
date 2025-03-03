device="${1:-1}"
num_shards=5
for (( i=$start ; i<$num_shards ; i++ ));
do
    CUDA_VISIBLE_DEVICES=$device python ../src/full_budget_train.py \
     --shard_id $i
done
