dataset_list=(game movie kindle)

for dataset in "${dataset_list[@]}"
do
    cd origin_data
    python amazon_process.py --dataset $dataset
    cd ..
    mkdir "data/${dataset}" "data/${dataset}/train" "data/${dataset}/test"
    cp "origin_data/${dataset}/10core/id2name.json" "data/${dataset}/"
    cp "origin_data/${dataset}/10core/id2name4Rec.json" "data/${dataset}/"
    cp "origin_data/${dataset}/10core/train.csv" "data/${dataset}/train/"
    cp "origin_data/${dataset}/10core/valid.csv" "data/${dataset}/train/"
    cp "origin_data/${dataset}/10core/test.csv" "data/${dataset}/test/"
    python process.py --process_type "data_process" --dataset $dataset
done
