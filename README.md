## Requirements
```
python>=3.7
numpy
torch
torchvision
pillow
tqdm
tensorboardX
scikit-learn
```
or
```
pip install -r requirements
```

## Download Trained Generative Models

```
wget https://www.dropbox.com/s/2426ilsu2jjuykf/params.zip
```
unzip and put `params` folder under `neural_compressor/model`

## NPC 
```
python main.py --dataset mnist --compressor gzip --data_dir data --distance NCD --shot 10
```

## NPC-LV
NPC-LV consists of 3 steps:
1. Train a latent variable generative model
2. Combined ANS with trained generative model to form a compressor
3. Use compressor and compressor-based distance metric for classification

### Step 2&3: Classification
This repo contains trained model parameters so we can do classification directly using the command below.
By default, only 100 test samples and 100 training samples will be used, which will take about half an hour to run.


```shell script
python main.py --compressor bbans --online --dataset mnist --shot 10
```


To use the same test indicies as the paper, which will include 1000 test samples, use `--replicate`. This process will require ~4 hours on 10-shot.

Below is the table of the result reported in the paper:

|         | MNIST  | FashionMNIST  | CIFAR-10 |
|:------: |:-----: |:-------------:| :-------:|
| 5-shot  | 77.6+-0.4 | 74.1+-3.2  | 35.3+-2.9|
| 10-shot | 84.6+-2.1 | 77.2+-2.2  | 36.0+-1.8|
| 50-shot | 91.4+-0.6 | 83.2+-0.7  | 37.4+-1.2|
| 100-shot| 93.6      |   84.5     |   40.2   |


### Step 1 (Optional): Train a Generative Model
```shell script
cd neural_compressor
python -m model.mnist_train --nz=2 --width=63
```
`neural_compressor` contains files for training a hierarchical VAE.
By default, trained generative model's parameters are saved under `neural_compressor/model/params/${dataset}/nz2`. 


## NPC-LV (large scale)
To replicate the experiment with 1000 test samples and 1000 training samples, we provide the compressed files to run knn directly, as compression takes a long time (1-2days).
Compressed files can be downloaded from [here](https://drive.google.com/file/d/1ftxLYbo3rBv1spGm1PB7hpZZfPdkGE0S/view?usp=sharing) (~2.1G).

After downloading and unzipping, pass compressed directories to command, e.g.,:

```shell script
python main.py --compressor bbans --dataset mnist --replicate --c_train_dir ../state_array/mnist/nz2/train1000 --c_test_dir ../state_array/mnist/nz2/test1000 --c_combined_dir ../state_array/mnist/nz2/avg/test1000_train1000
```  
Similarly, to replicate paper's result for fewer shot, add `--shot 10`.

 