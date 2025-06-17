from copy import deepcopy
from types import MethodType
from abc import ABC, abstractmethod
from typing import override

from nebula.config.config import Config
from nebula.core.models.nebulamodel import NebulaModel
from nebula.core.datasets.datamodule import DataModule
from nebula.core.unlearning.unlearningmanager import UnlearningMechanism

class UnlearningMethod(UnlearningMechanism, ABC):
    @abstractmethod
    def remove_unlearning_nodes(self, round: int) -> bool:
        pass


class ParameterResetting(UnlearningMethod):

    def __init__(
            self,
            model : NebulaModel,
            config=Config
            ):
        
        self.model = model
        self.initial_state_dict = deepcopy(model.state_dict())
        self.initial_learning_rate = model.learning_rate
        self.config = config
        self.unlearning_round = self.config.participant["unlearning_args"]["unlearning_round"]

    @override
    def remove_unlearning_nodes(self, round : int) -> bool:
        return round >= self.unlearning_round

    @override
    def before_training(self):
        self.model.load_state_dict(self.initial_state_dict)
        self.model.modify_learning_rate(self.initial_learning_rate)


class GradientAscent(UnlearningMethod):

    def __init__(
            self,
            model : NebulaModel,
            datamodule : DataModule,
            ):
        
        self.model = model
        self.datamodule = datamodule
        self.original_step = None
        self.original_weight = None
        self.unlearning_round = self.config.participant["unlearning_args"]["unlearning_round"]
        self.unlearning_rounds = self.config.participant["unlearning_args"]["unlearning_rounds"]
        self.unlearning_rounds = self.config.participant["unlearning_args"]["unlearning_rounds"]
        self.weight_factor = self.config.participant["unlearning_args"]["weight_factor"]

    @override
    def remove_unlearning_nodes(self, round : int) -> bool:
        return round >= self.unlearning_round + self.unlearning_rounds

    @override
    def before_training(self):
        self.original_step = self.model.step
        self.model.step = MethodType(self.model.step_gradient_ascent, self.model)

    @override
    def after_training(self):
        self.model.step = self.original_step
    
    @override
    def before_publishing(self):
        self.original_weight = deepcopy(self.datamodule.weight)
        self.datamodule.weight *= self.weight_factor

    @override
    def after_publishing(self):
        self.datamodule.weight = self.original_weight
