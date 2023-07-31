import numbers

from pydantic import Field, validator
from typing import List, Optional, Union, Any, Dict,Literal

from sdks.novavision.src.base.model import Package, Executor, ImageList, Param, Inputs, Configs, Outputs, Response, Request


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


class configTypeSegmentation(Param):
    name: Literal["segmentation"] = "segmentation"
    value: Literal["segmentation"] = "segmentation"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"



class ConfigType(Param):
    name: Literal["configType"] = "configType"
    value:Union[configTypeSegmentation]
    type: Literal["object"] = "object"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"



class SegmentationInputs(Inputs):
    inputImage: InputImage
    value: str
    type: Literal["object"] = "object"
    field: Literal["input"] = "input"


class SegmentationConfigs(Configs):
    configType: ConfigType
    value: str = "Configs"
    type: Literal["object"] = "object"
    field: Literal["config"] = "config"


class SegmentationOutputs(Outputs):
    OutputData: OutputData
    type: Literal["object"] = "object"
    field: Literal["output"] = "output"




class SegmentationRequest(Request):
    inputs: Optional[SegmentationInputs]
    configs: SegmentationConfigs
    class Config:
        schema_extra = {
            "target": "configs"
        }


class SegmentationResponse(Response):
    outputs: SegmentationOutputs



class SegmentationExecutor(Executor):
    name = "Segmentation"
    value: Union[SegmentationRequest, SegmentationResponse]
    type: Literal["Segmentation"] = "Segmentation"
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
    value: Union[SegmentationExecutor,TrainExecutor]
    type:Literal["executor"] = "executor"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"



class PackageModel(Package):
    type = "capsule"
    name = "Segmentation"
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
