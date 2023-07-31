import numbers

from pydantic import Field, validator
from typing import List, Optional, Union, Any, Dict,Literal

from capsules.TrafficSignClassify.src.models.Model import PackageExecutor  ## en son eklendi

from sdks.novavision.src.base.model import Package, Executor, ImageList, Param, Inputs, Configs, Outputs, Response, Request

#Yeni classlar oluşturuldu.

class ImageLabel(Param):
    name: Literal["ImageLabel"] = "ImageLabel"
    value: List
    type: Literal["ImageLabel"] = "ImageLabel"
    field: Literal["Label"] = "Label"

class OutputLabel(Param):
    name: Literal["OutputLabel"] = "OutputLabel"
    value: ImageLabel
    type: Literal["list"] = "list"
    field: Literal["Label"] = "Label"





class InputImage(Param):
    name: Literal["InputImage"] = "InputImage"
    value: ImageList
    type: Literal["imageList"] = "imageList"
    field: Literal["img"] = "img"

class ImageData(Param):
    name: Literal["Imagedata"] = "Imagedata"
    value: List
    type: Literal["Imagedata"] = "Imagedata"
    field: Literal["data"] = "data"

class OutputData(Param):
    name: Literal["OutputData"] = "OutputData"
    value: ImageData
    type: Literal["list"] = "list"
    field: Literal["data"] = "data"


class configTypeTraffic(Param):
    name: Literal["traffic"] = "traffic"
    value: Literal["traffic"] = "traffic"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"



class ConfigType(Param):
    name: Literal["configType"] = "configType"
    value:Union[configTypeTraffic]
    type: Literal["object"] = "object"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"



class TrafficInputs(Inputs):
    inputImage: InputImage
    value: str
    type: Literal["object"] = "object"
    field: Literal["input"] = "input"


class TrafficConfigs(Configs):
    configType: ConfigType
    value: str = "Configs"
    type: Literal["object"] = "object"
    field: Literal["config"] = "config"


class TrafficOutputs(Outputs):
    #OutputData: OutputData
    OutputLabel: OutputLabel
    type: Literal["object"] = "object"
    field: Literal["output"] = "output"

#OutputLabel eklendi.



class TrafficRequest(Request):
    inputs: Optional[TrafficInputs]
    configs: TrafficConfigs
    class Config:
        schema_extra = {
            "target": "configs"
        }


class TrafficResponse(Response):
    outputs: TrafficOutputs



class TrafficExecutor(Executor):
    name = "Traffic"
    value: Union[TrafficRequest, TrafficResponse]
    type: Literal["Traffic"] = "Traffic"
    field: Literal["executor"] = "executor"

    class Config:
        schema_extra = {
            "target": {
                "value": 0
            }
        }


class TrainOutputs(Outputs):
    OutputData: OutputData
    type: Literal["object"] = "object"
    field: Literal["output"] = "output"


class TrainResponse(Response):
    outputs: TrainOutputs

class BatchSize(Param):
    name: Literal["BatchSize"] = "BatchSize"
    value: int = Field(ge=1, le=100)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

class Path(Param):
    name: Literal["path"] = "path"
    value: str
    type: Literal["string"] = "string"
    field: Literal["textInput"] = "textInput"

class ConfigPath(Param):
    name: Literal["configPath"] = "configPath"
    value: Union[Path]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"


class ConfigBatchSize(Param):
    name: Literal["ConfigBatchSize"] = "ConfigBatchSize"
    value: Union[BatchSize]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"


class TrainConfigs(Configs):
    configPath: ConfigPath
    BatchSize: ConfigBatchSize
    value: str = "Configs"
    type: Literal["object"] = "object"
    field: Literal["config"] = "config"


class TrainRequest(Request):
    configs: TrainConfigs

    class Config:
        schema_extra = {
            "target": "configs"
        }


class TrainExecutor(Executor):
    name = "Train"
    value: Union[TrainRequest, TrainResponse]
    type: Literal["Train"] = "Train"
    field: Literal["executor"] = "executor"

    class Config:
        schema_extra = {
            "target": {
                "value": 0
            }
        }

class PackageExecutor(Executor):
    name = "executor"
    value: Union[TrafficExecutor,TrainExecutor]
    type:Literal["executor"] = "executor"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"


class PackageModel(Package):
    type = "capsule"
    name = "Traffic"
    uID = "1221112"
    executor: PackageExecutor
    field: Literal["executor"] = "executor"

    class Config:
        schema_extra = {
            "target": "executor"
        }


class RequestModel(Request):
    package: PackageModel


class ResponseModel(Response):
    package: PackageModel


