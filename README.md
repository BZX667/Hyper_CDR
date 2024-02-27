<<<<<<< HEAD
# Hyper_CDR
This repository contains a implementation of Hyper_CDR

## Environment Setup
1. Pytorch 2.0
2. Python 3.7
=======
# TaxoRec
This repository contains a implementation of our "Enhancing Recommendation with Automated TagTaxonomy Construction in Hyperbolic Space" accepted by ICDE 2022.

## Environment Setup
1. Pytorch 1.8.1
2. Python 3.7.3
>>>>>>> 48d15d6 (Add files via upload)

## Guideline

### data

<<<<<<< HEAD
Amazon CD
Amazon Movie
Amazon Book

### models

The implementation of model(```model.py```);  

code to implement Hyperbolic gcn (```encoders.py, hyp_layers.py```)  

### utils

```data_generator.py``` read and organize data  
```helper.py``` some method for helping preprocess data or set seeds and devices  
```sampler.py``` a parallel sampler to sample batches for training  
```taxogen.py``` build taxonomy  
```train_utils.py``` read and parse the config arguments  
=======
We provide one dataset, ciao.

```adj_csr.npz``` adj matrix built for training gcn 
```item_tag_matrix.npz``` items attributes matrix
```tag_map.json``` tag idx to tag name mapping.
```train.pkl``` train set
```test.pkl``` test set
```user_item_list.pkl``` user-item dict for the complete dataset.

### models

The implementation of model(```model.py```); 

code to implement Hyperbolic gcn (```encoders.py, hyp_layers.py```)

### utils

```data_generator.py``` read and organize data
```helper.py``` some method for helping preprocess data or set seeds and devices
```sampler.py``` a parallel sampler to sample batches for training
```taxogen.py``` build taxonomy
```train_utils.py``` read and parse the config arguments
>>>>>>> 48d15d6 (Add files via upload)

## Example to run the codes

```
python run.py
```

<<<<<<< HEAD
## Trans domain loss
set trans_loss=True in the config.py
=======
### Citation
If you find the code useful, please consider citing the following paper:
```
@inproceedings{tan2022enhancing,
  title={Enhancing Recommendation with Automated TagTaxonomy Construction in Hyperbolic Space},
  author={Tan, Yanchao and Yang, Carl and Wei, Xiangyu and Chen, Chaochao and Li, Longfei and Zheng, Xiaolin},
  booktitle={2022 IEEE 38th International Conference on Data Engineering (ICDE)},
  year={2022},
  organization={IEEE}
}
```
>>>>>>> 48d15d6 (Add files via upload)
