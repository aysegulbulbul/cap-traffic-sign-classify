# -*- coding: utf-8 -*-
"""Abstract base models"""

from abc import ABC, abstractmethod

from capsules.capsule.src.utils.config import Config


class BaseModel(ABC):
    """Abstract Model classes that is inherited to all models"""

    def __init__(self, cfg):
        self.config = Config.from_json(cfg)

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass