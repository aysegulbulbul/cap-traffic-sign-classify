import json

import tensorflow as tf
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))
from capsules.TrafficSignClassify.src.utils.config import Config
from capsules.TrafficSignClassify.src.configs.config import CFG
from capsules.TrafficSignClassify.src.models.PackageModel import TrafficExecutor,PackageModel,PackageExecutor,TrafficResponse,TrafficOutputs,ImageData,OutputData,OutputLabel,ImageLabel
from sdks.novavision.src.base.response import Response
from pydantic import ValidationError
from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.capsule import Capsule

class TrafficInferrer(Capsule):
    def __init__(self, request, bootstrap):
        self.error_list = []
        super().__init__(request)
        self.config = Config.from_json(CFG)
        self.image_size = self.config.data.image_size
        self.model =bootstrap["Traffic"]["model"]
        self.request.model = PackageModel(**(self.request.data))
        self.images = self.request.get_param("ImageList") #burasaı tetikleniyor.



    @staticmethod
    def bootstrap():#modeli kullanmak için
        config = Config.from_json(CFG)
        saved_path = config.project.path + '/capsules/capsule/src/weights/trafficsign/Trafic_signs_model.h5'
        model = tf.keras.models.load_model(saved_path)
        model = {"model":model}
        return model

    def preprocess(self, image):
        image = tf.image.resize(image, (self.image_size, self.image_size))
        return tf.cast(image, tf.float32) / 255.0

    def infer(self, image):
        tensor_image = tf.convert_to_tensor(image, dtype=tf.float32)
        tensor_image = self.preprocess(tensor_image)
        shape= tensor_image.shape
        tensor_image = tf.reshape(tensor_image,[1, shape[0],shape[1], shape[2]])
        print(tensor_image.shape)
        #pred = self.predict(tensor_image)
        #pred = pred.numpy().tolist()
        #pred=json.dumps(pred)
        #return json.loads(pred)

    @property
    def run(self):  #gelen parametreleri kullanıp cevap oluşturuyoruz(json tipinde)
            pred_list = []
           # img_list = []
            for img in self.images: #image ı yakaladik.
                pred=self.task(np.array(img.value)) #yakaladığım image oluşturduğum fonksiyona gönderdim
                pred_list.append(pred)
                #img_list.append(img)


            #imageList = ImageData(name="Imagedata", value=img_list, type="Imagedata",field="data")

            imageLabel=ImageLabel(name="ImageLabel",value=pred_list, type="ImageLabel", field="Label")
            outputLabel = OutputLabel(value=imageLabel, type="list",field="Label")


            #outputImage = OutputData(value=imageList)
            filterOutputs = TrafficOutputs(OutputLabel= outputLabel, name="Outputs", value="Outputs", type="object",field="output")

            filterResponse = TrafficResponse(outputs=filterOutputs)
            filterExecutor = TrafficExecutor(value=filterResponse)
            executor = PackageExecutor(value=filterExecutor)
            packageModel = PackageModel(executor=executor)
            return Response(model=packageModel).response()        #response tipinde cevap oluşturuyor.



    def task(self,image):

        image = tf.image.resize(image, (self.image_size, self.image_size))

        shape=image.shape
        tensor_image = tf.reshape(image, [1, shape[0], shape[1], shape[2]])
        pred = np.argmax(self.model.predict([tensor_image])).astype("float32")

        #self.model=h5 ->tahmin yapmasını sağlıyor =pred

        classes = {1: 'Speed limit (20km/h)',
                   2: 'Speed limit (30km/h)',
                   3: 'Speed limit (50km/h)',
                   4: 'Speed limit (60km/h)',
                   5: 'Speed limit (70km/h)',
                   6: 'Speed limit (80km/h)',
                   7: 'End of speed limit (80km/h)',
                   8: 'Speed limit (100km/h)',
                   9: 'Speed limit (120km/h)',
                   10: 'No passing',
                   11: 'No passing veh over 3.5 tons',
                   12: 'Right-of-way at intersection',
                   13: 'Priority road',
                   14: 'Yield',
                   15: 'Stop',
                   16: 'No vehicles',
                   17: 'Veh > 3.5 tons prohibited',
                   18: 'No entry',
                   19: 'General caution',
                   20: 'Dangerous curve left',
                   21: 'Dangerous curve right',
                   22: 'Double curve',
                   23: 'Bumpy road',
                   24: 'Slippery road',
                   25: 'Road narrows on the right',
                   26: 'Road work',
                   27: 'Traffic signals',
                   28: 'Pedestrians',
                   29: 'Children crossing',
                   30: 'Bicycles crossing',
                   31: 'Beware of ice/snow',
                   32: 'Wild animals crossing',
                   33: 'End speed + passing limits',
                   34: 'Turn right ahead',
                   35: 'Turn left ahead',
                   36: 'Ahead only',
                   37: 'Go straight or right',
                   38: 'Go straight or left',
                   39: 'Keep right',
                   40: 'Keep left',
                   41: 'Roundabout mandatory',
                   42: 'End of no passing',
                   43: 'End no passing veh > 3.5 tons'}

        sign = classes[pred + 1]
        algilanan = {"alginlanan_tip": sign}  # "image": image.numpy().tolist()}
        algilanan = json.dumps(algilanan)
        return algilanan
        #return json.loads(algilanan)
        #print("Hello")
        print(sign)
        #print('Hello')

#modeli içeri attım, executorsa modeli kullanarak verdiğim görüntüyü işledim,
# daha sonra bunlara cevap oluşturup geri döndürdüm.


