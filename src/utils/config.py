# -*- coding: utf-8 -*-
"""Config classes"""

import json


class Config:
    """Config classes which contains data, train and models hyperparameters"""

    def __init__(self, data, train, model, project):
        self.data = data
        self.train = train
        self.model = model
        self.project = project

    @classmethod
    def from_json(cls, cfg):
        """Creates configs from json"""
        params = json.loads(json.dumps(cfg), object_hook=HelperObject)
        return cls(params.data, params.train, params.model, params.project)


class HelperObject(object):
    """Helper classes to convert json into Python object"""
    def __init__(self, dict_):
        self.__dict__.update(dict_)