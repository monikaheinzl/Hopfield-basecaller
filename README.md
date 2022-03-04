# Basecalling of Nanopore sequencing data using modern Hopfield networks
This repository contains the source code of my [master thesis](https://epub.jku.at/obvulihs/content/titleinfo/6966694) supervised by Prof. Sepp Hochreiter (Institute for Machine Learning), Prof. Irene Tiemann-Boege (Insitute of Biophysics) and Bernhard Schäfl, MSc. (Institute for Machine Learning) at the Johannes Kepler University Linz, Austria. 

## Dependencies
* python3
* pytorch_gpu 1.6
* [hopfield-layers](https://github.com/ml-jku/hopfield-layers)
* [ctcdecode](https://github.com/parlance/ctcdecode.git) for Architecture C

The file condaenv.yml contains all conda packages necessary for the execution of training and basecalling scripts. Note that different architectures were tested during the thesis and thus, some of the packages are not needed for the execution of the final architecture. 

## Usage
### Preprocessing
The architecture is trained on bacterial and viral data published by the basecaller [Chiron](https://github.com/haotianteng/Chiron). 

#### Training a model
In this thesis, we tested 3 different architectures (see also Figure 11), which can be trained by the following scripts. All of them are set with the final hyperparameters as in Table 4 of the thesis. 
##### Architecture A:
It consists of an encoder-decoder architecture using CNN-LSTM layers during encoding and LSTM layers followed my a fully-connected layer for decoding and can be run by

```angular2
$ sh train_LSTM.sh
```

##### Architecture B:
It consists of an encoder-decoder architecture using CNN and modern Hopfield layers during encoding and modern Hopfield layers followed my a fully-connected layer for decoding and can be run by

```angular2
$ sh train_Hopfield.sh
```
This architecture shows the best results on our validation set and is therefore used for inference.


##### Architecture C:
It differs from Architecture by not using the modern Hopfield networks as a decoder but the CTC algorithm and can be run by

```angular2
$ sh train_Hopfield_CTC.sh
```

For all architectures the following parameters should be modified when training:
`infile`: pickle file that is created in the preprocessing step with the training, validation (and testing) set.
`outfolder`: name of the folder that is created after training and contains the following output files: 
txt file with the model's summary (outfolder name.txt), event file of TensorBoard to track the loss, edit distance (and accuracy), json file with the hyperparameters of the model (model.json) and a pickle file with the learned model (outfolder name.pt). The latter two files are required for inference. 

### Basecalling
#### Preprocessing
#### Basecalling

### Analysis
#### Preprocessing
#### Basecalling