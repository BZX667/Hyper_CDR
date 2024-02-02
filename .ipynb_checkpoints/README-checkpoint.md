# Hyper_CDR
This repository contains a implementation of Hyper_CDR

## Environment Setup
1. Pytorch 1.8.1
2. Python 3.7.3

## Guideline

### data

Amazon CD
Amazon Movie
Amazon Book

### models

The implementation of model(```model.py```); 

code to implement Hyperbolic gcn (```encoders.py, hyp_layers.py```)

### utils

```data_generator.py``` read and organize data \n
```helper.py``` some method for helping preprocess data or set seeds and devices \n
```sampler.py``` a parallel sampler to sample batches for training \n
```taxogen.py``` build taxonomy \n
```train_utils.py``` read and parse the config arguments \n

## Example to run the codes

```
python run.py
```

## Trans domain loss
set trans_loss=True in the config.py
