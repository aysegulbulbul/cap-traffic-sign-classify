import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../../../'))
import requests
import cv2
import numpy as np
import json
from sdks.novavision.src.media.image import Image as image

from sdks.novavision.src.base.model import Image, ImageList, Request
from capsules.TrafficSignClassify.src.utils.config import Config
from capsules.TrafficSignClassify.src.configs.config import CFG

from capsules.TrafficSignClassify.src.models.PackageModel import PackageModel,PackageExecutor,TrafficConfigs,TrafficInputs,TrafficExecutor,TrafficRequest,ConfigType,InputImage,configTypeTraffic
ENDPOINT_URL = "http://127.0.0.1:8000/api"

#web tarafına ihtiyaç duymadan test etmek = Çıkarım dosyası

def save_json_to_file(json_data, filename): #Json verisini bir dosyaya yazar.
    with open(filename, 'w') as f:
        f.write(json.dumps(json_data)) #Json_data sözlüğünü JSON biçimine dönüştürür.


def inference():
    config = Config.from_json(CFG)
    image_data =Image(name="image", uID="323332", mime_type="image/png", encoding="base64",value =image.encode64(np.asarray(cv2.imread(config.project.path +'/capsules/TrafficSignClassify/resources/00000.png')).astype(np.float32),'image/png'), type="imageList", field="img")

    traffic = configTypeTraffic(value="traffic")
    configTypevalue = ConfigType(value=traffic)
    trafficConfigs = TrafficConfigs(configType=configTypevalue, name="Configs")#

    imageList = ImageList(name="ImageList", value=[image_data], type="imageList", field="img")
    inputImage = InputImage(value=imageList)
    trafficInputs = TrafficInputs(inputImage=inputImage, name="Inputs", value="Inputs")
    trafficRequest = TrafficRequest(inputs=trafficInputs, configs=trafficConfigs)
    trafficExecutor = TrafficExecutor(value=trafficRequest)
    executor = PackageExecutor(value=trafficExecutor)
    request = PackageModel(executor=executor, name="Traffic")
    request_json = json.loads(request.json())

    #JSON verilerini resources klasöründeki bir dosyaya kaydediliyor.
    save_json_to_file(request_json,os.path.join(config.project.path, 'capsules/TrafficSignClassify/resources', 'request_data.json'))

    response = requests.post(ENDPOINT_URL, json =request_json) #post isteği atıyor.
    #print(response.raise_for_status())
    #print(response.json())  #cevap
    print(json.dumps(request_json))
    #print(request.json()) #rica etmek


if __name__ =="__main__":
    inference()