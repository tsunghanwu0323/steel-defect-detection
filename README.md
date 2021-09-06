# steel-defect-detection
# [Severstal: Steel Defect Detection](https://www.kaggle.com/c/severstal-steel-defect-detection)

A image classification and segmentation task,  please download the .csv files and put them into a ./data folder in your local enviroment. 

## Generate Config
Supported model: DenseNet, CNN and VGG
```
himl hiera/classification/model="model name"/ --output-file config/config.yaml
```

## Run 
```
python run.py --config config/config.yaml 
```
