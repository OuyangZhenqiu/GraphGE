# GraphGE

## Step0. Install Libs:
```
conda create --name graphGE python=3.6
conda activate graphGE
conda install -c rdkit rdkit
pip install fitlog
pip install matplotlib
pip install scikit-learn (0.24.2)
pip install scipy (1.5.4)
pip install torch (1.10.2+cu113)
pip install torch-cluster (1.5.9) (https://pytorch-geometric.com/whl/)  
pip install torch-scatter (2.0.9) (https://pytorch-geometric.com/whl/)  
pip install torch-sparse (0.6.12) (https://pytorch-geometric.com/whl/)  
pip install torch-spline-conv (1.2.1) (https://pytorch-geometric.com/whl/)  
pip install torch-geometric (2.0.2) (https://pytorch-geometric.com/whl/)  
```
## Step1: Data Preprocessing
To download the data from `https://drive.google.com/drive/folders/1bgq0gQIzT4GoEfDj_Z9ZeYmXI873gTDm` and unzip it into the folder '/data'.

To use the file `preprocess_data.ipynb` to preprocess the data to obtain GDSC and TCGA dataset.


## Step2: Train and evaluate model
To train and evaluate Graph on the GDSC dataset, run:
```
python multi_label_GDSC_GE.py
```
To train and evaluate Graph on the TCGA dataset, run:
```
python multi_label_TCGA_GE.py
```