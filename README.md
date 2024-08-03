# Addressing Personalized Bias for Unbiased Learning to Rank

This repository contains the supporting code for our paper "Addressing Personalized Bias for Unbiased Learning to Rank".
The implementation for training and evaluating ULTR models is based on the [ULTRA_pytorch](https://github.com/ULTR-Community/ULTRA_pytorch) toolkit.

## Repository Usage
### Get Started
```
pip install -r requirements.txt
```

### Preparing dataset
In our paper, we utilized two public datasets: [Yahoo! LETOR set 1](https://webscope.sandbox.yahoo.com/catalog.php?datatype=c) and [Baidu-ULTR](https://github.com/ChuXiaokai/baidu_ultr_dataset). For reproducibility, one can download 
the datasets and convert the data into the "ULTRA" form (containing "xxx.feature", "xxx.init_list", and "xxx.labels" three files) as shown in the "example_dataset" fold.

### Click simulation
To obtain training ranked lists with user IDs:
```
python allocate_user.py [TRAINING_LABEL_PATH] [USER_NUM] [SESSION_NUM] [INITIAL_RANKED_LIST_PATH]
```

To generate click data on the obtained training ranked lists:
```
python generate_labels_PBM.py [RANKEDLIST_PATH] [TRAINING_LABEL_PATH] [USER_NUM]
```

### Training and evaluating ULTR models
This project implements our user-aware estimator and reproduces the offline ULTR models based on [ULTRA_pytorch](https://github.com/ULTR-Community/ULTRA_pytorch). Please see [ULTRA_pytorch](https://github.com/ULTR-Community/ULTRA_pytorch) for more details about this framework. Here we give the simple instructions to reproduce our experiments.

To estimate relevance labels from clicks using our user-aware estimator:
```
python relevance_estimator.py [TRAINING_LABEL_PATH] [USER_NUM]
```

To train an MLP ranking model with the relevance labels estimated by our user-aware estimator:
```
python main.py --data_dir [DATASET_PATH] --train_dataset train --train_data_prefix user_aware --model_dir [SAVE_MODEL_PATH] --setting_file ./example_settings/naive_exp_settings.json
```

To directly train a ULTR model like DLA using clicks:
```
python main_click.py --ULTR_model DLA --data_dir [DATASET_PATH] --train_dataset train_click --train_data_prefix train --model_dir [SAVE_MODEL_PATH] --setting_file ./example_settings/naive_exp_settings.json
```
To evaluate a ULTR method (with an MLP ranking model):
```
python main.py --data_dir [DATASET_PATH]  --setting_file ./example_settings/test_exp_settings.json --batch_size 1 --test_only True  --model_dir [MODEL_PATH]
```