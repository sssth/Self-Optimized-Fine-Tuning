# SOFT

This is the implementation of our paper: "Bridging the Gap: Self-Optimized Fine-Tuning for LLM-based Recommender Systems"

#### Data processsing
We utilize the Amazon Review dataset, which can be downloaded from the following URL:

https://amazon-reviews-2023.github.io/

We utilized only the following two files from the Amazon Review dataset, which should be placed at "./origin_data/{dataset}":
* "interaction.csv" contains user-item interaction history information, specifically focusing on ratings only.
* "metadata.jsonl" encompasses all items along with their corresponding names.
Next, execute the following commands to preprocess the data:
```
bash process.bash
```

#### Training
To run SFT and SOFT with BIGRec as the backbone, execute the following commands:
```
bash train_BIGRec.bash
```
To run SFT and SOFT with LLaRA as the backbone, execute the following commands:
```
bash train_BIGRec.bash
```