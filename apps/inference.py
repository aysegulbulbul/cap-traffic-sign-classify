import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../../../'))
import requests
import cv2
import numpy as np
import json
from sdks.novavision.src.media.image import Image as image

from sdks.novavision.src.base.model import Image, ImageList, Request
from capsules.capsule.src.utils.config import Config
from capsules.capsule.src.configs.config import CFG

from capsules.capsule.src.models.PackageModel import PackageModel,PackageExecutor,SegmentationConfigs,SegmentationInputs,SegmentationExecutor,SegmentationRequest,ConfigType,InputImage,configTypeSegmentation


ENDPOINT_URL = "http://127.0.0.1:8000/api"


def inference():
    config = Config.from_json(CFG)
    image_data =Image(name="image", uID="323332", mime_type="image/jpg", encoding="base64",value =image.encode64(np.asarray(cv2.imread(config.project.path +'/capsules/capsule/resources/yorkshire_terrier.jpg')).astype(np.float32),'image/jpg'), type="imageList", field="img")
    segmentation = configTypeSegmentation(value="segmentation")
    configTypevalue = ConfigType(value=segmentation)
    segmentationConfigs = SegmentationConfigs(configType=configTypevalue, name="Configs")
    imageList = ImageList(name="ImageList", value=[image_data], type="imageList", field="img")
    inputImage = InputImage(value=imageList)
    segmentationInputs = SegmentationInputs(inputImage=inputImage, name="Inputs", value="Inputs")
    segmentationRequest = SegmentationRequest(inputs=segmentationInputs, configs=segmentationConfigs)
    segmentationExecutor = SegmentationExecutor(value=segmentationRequest)
    executor = PackageExecutor(value=segmentationExecutor)
    request = PackageModel(executor=executor, name="Segmentation")
    request_json = json.loads(request.json())
    response = requests.post(ENDPOINT_URL, json =request_json)
    print(response.raise_for_status())
    print(response.json())



if __name__ =="__main__":
    inference()