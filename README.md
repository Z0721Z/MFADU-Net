# MFADU-Net

## Requirements
1. pytorch==1.10.0
2. pytorch-lightning==1.1.0
3. albumentations==0.3.2
4. seaborn
5. sklearn
## Dataset
To apply the model on a custom dataset, the data tree should be constructed as:
``` 
    ├── data
          ├── images
                ├── image_1.png
                ├── image_2.png
                ├── image_n.png
          ├── masks
                ├── image_1.png
                ├── image_2.png
                ├── image_n.png
```
## CSV generation 
```
python data_split_csv.py --dataset your/data/path --size 0.9 
```
## Train
```
python train.py --dataset your/data/path --csvfile your/csv/path --loss dice --batch 16 --lr 0.001 --epoch 150 
```
## Evaluation
```
python eval_binary.py --dataset your/data/path --csvfile your/csv/path --model save_models/epoch_last.pth --debug True
```
