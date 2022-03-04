# Basecalling of Nanopore sequencing data using modern Hopfield networks
This repository contains the source code of my [master thesis](https://epub.jku.at/obvulihs/content/titleinfo/6966694) supervised by Prof. Sepp Hochreiter (Institute for Machine Learning), Prof. Irene Tiemann-Boege (Insitute of Biophysics) and Bernhard Schäfl, MSc. (Institute for Machine Learning) at the Johannes Kepler University Linz, Austria. 

## Dependencies
* python3
* pytorch_gpu 1.6
* [hopfield-layers](https://github.com/ml-jku/hopfield-layers)
* [ctcdecode](https://github.com/parlance/ctcdecode.git) for architecture C
* [Transformer-pytorch](https://github.com/dreamgonfly/Transformer-pytorch) was partially adapted and is now contained in the directory `basecaller-modules`
The file condaenv.yml contains all conda packages necessary for the execution of training and basecalling scripts. Note that different architectures were tested during the thesis and thus, some of the packages are not needed for the execution of the final architecture. 

## Usage
### Training
#### Preprocessing

The architecture is trained on bacterial and viral data published by the basecaller [Chiron](https://github.com/haotianteng/Chiron). It provides the raw signals in `.signal` and the corresponding DNA sequences in `.label` files. 
First, the data is read into a dictionary storing the raw signal, labels (DNA sequence), start and end coordinates of the segments in the raw signal that encode to one nucleotide (script `preprocessing/read_allData_to_dict.py`). In the thesis, we cluster the signal data set by spectral clustering into training, validation and test set based on distances calculated by Dynamic Time Warping (DTW). The class labels are stored in a separate file and are then used in the script `preprocessing/split_into_windows.py` for splitting the raw signals of the the training, validation and test set into windows. The default window size is `2048`.
For architecture C the preprocessing is done with the script `preprocessing/split_into_windows_CTC.py` as it does not require an `<EOS>` token. 

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

###### Architecture C:
It differs from Architecture by not using the modern Hopfield networks as a decoder but the CTC algorithm and can be run by

```angular2
$ sh train_Hopfield_CTC.sh
```

For all architectures the following parameters should be modified during training:
`infile`: pickle file that is created in the preprocessing step with the training, validation (and testing) set.
`outfolder`: appendix of the folder name that is created after training (training_result_ + outfolder name)

The ouput are the following files: 
txt file with the model's summary (outfolder name.txt), event file of TensorBoard to track the loss, edit distance (and accuracy), PDF file with loss, edit distance (and accuracy) of both training and validation set, json file with the hyperparameters of the model (model.json) and a pickle file with the learned model (outfolder name.pt). The latter two files are required for inference. 

### Basecalling
Architecture B shows the best results on our validation set and therefore, the basecalling is 


provide the trained model `training_result_final_model/final_model.pt` with its hyperparameter settings `training_result_final_model/model.json`

```angular2
$ sh inference_Hopfield.sh
```

The following parameters should be modified during inference:
`infile`: directory containing FAST5 files 
`modelfolder`: file of the trained model (e.g. `training_result_final_model/final_model.pt`)
`configfile` (optional): file with hyperparameters of the trained model (e.g. `training_result_final_model/model.json`). If parameter is `None`, the hyperparameters can be defined in the script `inference_Hopfield.sh`
`outfolder`: name of the output directory. 

The output are the following files: 
A directory containing the FASTA files with the prediction for each window of a read and a FASTA file with the completely predicted read sequence (concatenation of the predicted windows). 