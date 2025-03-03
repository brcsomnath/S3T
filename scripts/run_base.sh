# 1: device, 2: dataset, 3: benchmark, 4: rank, 5: lr, 6: stop_epochs, 7: save_steps
device="${1:-0}"
dataset="${2:-qnli}"
benchmark="${3:-glue}"
rank="${4:-16}"
lr="${5:-1e-4}"
stop_epochs="${6:-30}"
save_steps="${7:-1000}"
num_slices="${8:-7}"

echo "GPU Device ID: "$device""
echo "Dataset: "$dataset""
echo "Benchmark: "$benchmark""
echo "Rank: "$rank""
echo "Learning rate: "$lr""
echo "Stop Epochs: "$stop_epochs""
echo "Save Steps: "$save_steps""

for (( i=$start ; i<$num_slices ; i++ )); 
do
  CUDA_VISIBLE_DEVICES=$device python ../src/s3t_base.py --dataset $dataset \
   --benchmark $benchmark \
   --rank $rank \
   --lr $lr \
   --shard_id $i \
   --stop_epochs $stop_epochs \
   --save_steps $save_steps
done