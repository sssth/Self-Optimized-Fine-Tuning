dataset_list=(game movie kindle)               
CUDA_DEVICES="4,5,6,7"
main_process_port=29512
base_model_path="../Llama3.2-3B"



last_cuda=$(echo $CUDA_DEVICES | awk -F',' '{print $NF}')
for dataset in "${dataset_list[@]}"
do  
    # encode item name
    CUDA_VISIBLE_DEVICES=$last_cuda accelerate launch encode.py --dataset_name $dataset --base_model $base_model_path

    # run and evaluate SFT
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICES accelerate launch --main_process_port $main_process_port train_llara.py \
        --train_type "SFT" --dataset_name $dataset --base_model $base_model_path
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICES accelerate launch --main_process_port $main_process_port inference_llara.py  \
        --method "llara-SFT" --inference_type "test" --dataset $dataset
    CUDA_VISIBLE_DEVICES=$last_cuda accelerate launch evaluate_batch.py \
        --method "llara-SFT" --inference_type "test" --dataset $dataset
    
    # generate SD dataset
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICES accelerate launch --main_process_port $main_process_port inference_llara.py \
        --method "llara-SFT" --inference_type "generate" --dataset $dataset
    CUDA_VISIBLE_DEVICES=$last_cuda accelerate launch evaluate_batch.py  \
        --method "llara-SFT" --inference_type "generate" --dataset $dataset
    python process.py --dataset $dataset --method "llara-SFT" --process_type "mindist"

    # run and evaluate SOFT
    alpha_list=(100 10 1 0.1)
    for alpha in "${alpha_list[@]}"
    do
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICES accelerate launch --main_process_port $main_process_port train_llara.py \
        --train_type "SOFT" --alpha $alpha --dataset_name $dataset --base_model $base_model_path
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICES accelerate launch --main_process_port $main_process_port inference_llara.py \
        --method "llara-SOFT-${alpha}" --inference_type "test" --dataset $dataset
    CUDA_VISIBLE_DEVICES=$last_cuda accelerate launch evaluate_batch.py \
        --method "llara-SOFT-${alpha}" --inference_type "test" --dataset $dataset
    done
    
done
