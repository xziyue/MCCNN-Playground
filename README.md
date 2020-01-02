# Playground

Playing with the code and data of the following article ([repo](https://bitbucket.org/mkoohim/multichannel-cnn/src/master/)):

Koohi-Moghadam, Mohamad, et al. "Predicting disease-associated mutation of metal-binding sites in proteins using a deep learning approach." *Nature Machine Intelligence* (2019): 1-7.


Visualizations/results are shown in Jupyter Notebooks in the `notebook` folder.

## Startup sequence

One must extract the features from original dataset first. To do so, run the following scripts
in the correct order. 

1. run `./source/process_spatial_data.py`
2. run `./source/construct_training_data.py`

