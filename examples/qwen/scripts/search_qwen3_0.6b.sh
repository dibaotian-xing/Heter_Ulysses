export CLUSTER_TYPE='a6000x2_id12'
export model_name='qwen3_0.6b'
export seq_length=4096
export num_query_groups=8
export batch_size=4
export num_hidden_layers=32
export num_attention_heads=32
export head_dim=128
export gpu_num_list=(1 1) #gpu nums for different gpu types
export gpu_capacity_list=(48 48) #gpu capacity for different gpu types
export model_parameter_size=0.6
export precision='fp16'

python ipalg/search.py \
    --cluster_type $CLUSTER_TYPE \
    --model_name $model_name \
    --seq_length $seq_length \
    --num_query_groups $num_query_groups \
    --batch_size $batch_size \
    --num_hidden_layers $num_hidden_layers \
    --num_attention_heads $num_attention_heads \
    --head_dim $head_dim \
    --gpu_type_num_list "${gpu_num_list[@]}" \
    --gpu_type_mem_capacity_list "${gpu_capacity_list[@]}" \
    --model_parameter_size $model_parameter_size \
    --precision $precision
