# Basecalling of Nanopore sequencing data using modern Hopfield networks
This repository contains the source code of my [master thesis](https://epub.jku.at/obvulihs/content/titleinfo/6966694) supervised by Prof. Sepp Hochreiter (Institute for Machine Learning), Prof. Irene Tiemann-Boege (Insitute of Biophysics) and Bernhard Schäfl, MSc. (Institute for Machine Learning) at the Johannes Kepler University Linz, Austria. 

## Dependencies
* python3
* pytorch_gpu 1.6
* [hopfield-layers](https://github.com/ml-jku/hopfield-layers)
* [ctcdecode](https://github.com/parlance/ctcdecode.git) for architecture C
* [Transformer-pytorch](https://github.com/dreamgonfly/Transformer-pytorch) is partially adapted and is now contained in the directory `basecaller-modules`.

The file condaenv.yml contains all conda packages necessary for the execution of training and basecalling scripts. Note that different architectures were tested during the thesis and thus, some of the packages are not needed for the execution of the final architecture. 

## Usage
### Training
#### Preprocessing
The architectures are trained on bacterial and viral data published by the basecaller [Chiron](https://github.com/haotianteng/Chiron). It provides the raw signals in `.signal` files and the corresponding DNA sequences in `.label` files. We read the files into a dictionary storing the raw signal, labels (DNA sequence), start and end coordinates of the segments in the raw signal that encode to one nucleotide (script `preprocessing/read_allData_to_dict.py`). 
This dictionary is then the input for the preprocessing step if you want to train the architecture with your own data. In the thesis, we clustered the data from this dictionary into training, validation and test set. Therefore, we provide also a `npz` file with class labels of the resulting clusters. The architectures are then trained on windows of the raw signal (default window size is `2048`) and segmentation can be done with the scripts `preprocessing/split_into_windows.py` (architecture A and B) and `preprocessing/split_into_windows_CTC.py` (architecture C). The resulting `pickle` file contains then the windows and labels for your training and validation set, which is then the input for the scripts in the next section. 

#### Training a model
In this thesis, we tested 3 different architectures (see also Figure 11), which can be trained by the following scripts. All of them are set with the final hyperparameters as in Table 4 of the thesis. 
##### Architecture A:
It consists of an encoder-decoder architecture using CNN-LSTM layers during encoding and LSTM layers followed by a fully-connected layer for decoding and can be run by

```angular2
$ sh train_LSTM.sh
```
##### Architecture B:
It consists of an encoder-decoder architecture using CNN and modern Hopfield layers during encoding and modern Hopfield layers followed by a fully-connected layer for decoding and can be run by

```angular2
$ sh train_Hopfield.sh
```
##### Architecture C:
It differs from architecture B by not using the modern Hopfield networks as a decoder but the CTC algorithm and can be run by

```angular2
$ sh train_Hopfield_CTC.sh
```

For all architectures the following parameters should be modified during training:
* `infile`: pickle file that is created in the preprocessing step
* `outfolder`: appendix of the folder name that is created after training (training_result_ + `outfolder` name)

The directory `outfolder` will then contain the following files after training: 
* txt file with the model's summary (`outfolder` name.txt)
* event file of TensorBoard to track the loss, edit distance (and accuracy) during training
* PDF file with loss, edit distance (and accuracy) after training finishes 
* json file with the hyperparameters of the model (model.json) 
* pickle file with the learned model (`outfolder` name.pt)
The latter two files are required for the basecalling. 

### Basecalling
We provide the trained model `training_result_final_model/final_model.pt` with its hyperparameter settings `training_result_final_model/model.json` for architecture B. 

```angular2
$ sh inference_Hopfield.sh
```

The following parameters should be modified during inference:
* `infile`: directory containing FAST5 files
* `modelfolder`: file of the trained model (e.g. `training_result_final_model/final_model.pt`)
* `configfile` (optional): file with hyperparameters of the trained model (e.g. `training_result_final_model/model.json`). If parameter is `None`, the hyperparameters can be defined in the script `inference_Hopfield.sh`
* `outfolder`: name of the output directory

The output are the following files: 
* directory containing the FASTA files with the prediction for each window of a read
* FASTA file with the completely predicted read sequence (concatenation of the predicted windows)