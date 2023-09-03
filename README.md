# TrafficSignClassify Capsule for NovaVision

## Requirements
- FastApi Service  
- Docker
- GitSCM

## Installation
1. Capsules should be under capsules directory
```
git submodule add https://github.com/novavision-ai/package.git TrafficSignClassify
```
2. After this step, modify the "executor" information in the "service.py" file inside the image. Change the "Executor" key value to "Traffic" and set the value of the "Traffic" key to "TrafficInferrer":
```
executors = {'Traffic': {"Traffic": TrafficInferrer}}
```
3. In the project directory, open a terminal and navigate to the "apps" directory under the "TrafficSignRecognition" capsule. Use the following command to perform inference:
```
python inference.py
```
The output should be in JSON format, printed on the terminal screen, and the json must include the class of the given input image.


## Directory Structure
```
apps
   |-- inference.py
   |-- train.py
notebooks
   |-- unet.ipynb
   |-- TrafficSignModalTrain.ipynb
resources
   |-- request_data.json
   |-- yorkshire_terrier.jpg
setup.py
src
   |-- __init__.py
   |-- classes
   |   |-- base_class.py
   |-- configs
   |   |-- config.py
   |   |-- data_schema.py
   |   |-- logging_config.yaml
   |-- dataloaders
   |   |-- dataloader.py
   |-- executors
   |   |-- segmentation.py
   |   |-- trafficsign.py
   |   |-- trainer.py
   |-- models
   |   |-- PackageModel.py
   |   |-- train_model.py
   |   |-- u_net_model.py
   |-- utils
   |   |-- config.py
   |   |-- logger.py
   |   |-- plot_image.py
   |-- weights
       |--trafficsign
           |--Trafic_signs_model.h5
   |   |-- unet
   |   |   |-- saved_model.pb
   |   |   |-- variables
   |   |   |   |-- variables.data-00000-of-00001
   |   |   |   |-- variables.index
tests
   |-- unet_inferrer.py
```

## Functionality of the Used Files in the Capsule:

* Apps:
  * inference.py: Used to create a JSON request using classes defined in PackageModel.
* Notebooks:
  * YoloModel.ipynb: Contains python code for local model training.
* Resources: The directory for input images used while creating a request within the inference file.
* src:
  * Configs:
    * config.py: Contains config files that includes options to make changes on inputs.
  * Executors:
    * trafficsign.py: Used to create a JSON response using classes defined in PackageModel. 
  * Models:
    * PackageModel.py: Contains class definitions for creating JSON requests and responses using the pydantic module.
  * Weights:
    * Trafic_signs_model.h5: We used the weight of the model we trained in colab.  


## Capsule Workflow

The Trafic_signs_model weight file used in the TrafficSignClassify capsule is trained with 31375 images of different sizes. In inference, the JSON request is created using the classes in the packagemodel. The place where JSON object classes are written in the package model is referred to as the "package model" itself.This request is then sent to the trafficsign executor file. The trafficsign executor file generates a response JSON in response to the request.



