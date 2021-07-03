# README

Code for argument classification models generation.

## Dependencies

-Pytorch 

-Numpy

-Pandas

-Scikit-learn

-nltk

## Data and resources

To train these models, we used AMPERE dataset in [hua-etal-2019-argument-mining](https://www.aclweb.org/anthology/N19-1219/). You need to download the [dataset](http://xinyuhua.github.io/Resources/naacl19/) if you want to retrain the models.

The preteained SciBert model is required.

```
wget -O scibert_uncased.tar https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/huggingface_pytorch/scibert_scivocab_uncased.tar
tar -xvf scibert_uncased.tar
```

## Running

First, please run the code to get the data set:

```
python get_dataset.py
```

If you want to get baseline models (SVM and MLP):

```
python baseline.py
```

If you want to get other models (Bert and SciBert):

```
python models.py
```



