# -*- coding: utf-8 -*-
""" train.py """
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../../../'))
import requests
import json
from capsules.capsule.src.models.Model import PackageModel,PackageExecutor,TrainConfigs,TrainExecutor,TrainRequest,ConfigPath,ConfigBatchSize,Path,BatchSize
from capsules.capsule.src.utils.config import Config
from capsules.capsule.src.configs.config import CFG


ENDPOINT_URL = "http://127.0.0.1:8000/api"


def train():
    config = Config.from_json(CFG)
    path =Path(value=config.data.path)
    batchSize=BatchSize(value=64)
    configPath=ConfigPath(value=path)
    configBathSize =ConfigBatchSize(value=batchSize)
    trainConfigs =TrainConfigs(BatchSize=configBathSize,configPath=configPath,name="Configs")
    trainRequest = TrainRequest(configs=trainConfigs)
    trainExecutor = TrainExecutor(value=trainRequest)
    executor = PackageExecutor(value=trainExecutor)
    request = PackageModel(executor=executor, name="Segmentation")
    request_json = json.loads(request.json())
    response = requests.post(ENDPOINT_URL, json=request_json)
    print(response.raise_for_status())
    print(response.json())


if __name__ == '__main__':
    train()