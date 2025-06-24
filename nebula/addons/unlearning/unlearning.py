import logging
from copy import deepcopy
from abc import ABC

from nebula.core.models.nebulamodel import NebulaModel
from nebula.core.datasets.datamodule import DataModule
from nebula.core.training.lightning import Lightning
from nebula.addons.unlearning.extension import LearningCycleExtension


class UnlearningCycleExtension(LearningCycleExtension, ABC):
    """
    Base class for unlearning cycle extensions. This class should
    do nothing when set as active extension and serve as a default.
    """
    def remove_passive_nodes(self, round: int) -> bool:
        return False


class ParameterResetting(UnlearningCycleExtension):
    """
    This extension handles basic retraining by resetting the model parameters
    and learning rate to their initial values at the specified unlearning round.
    An random training step is set to avoid immediate retraining after resetting
    to be able to compare it to other unlearning methods.
    """
    def __init__(
            self,
            model : NebulaModel,
            unlearning_params : dict,
            ):
        self.model = model
        self.unlearning_round: int = unlearning_params["unlearning_round"]
        self.initial_state_dict = deepcopy(model.state_dict())
        self.initial_learning_rate = model.learning_rate
        self.previous_training_step = None

    def remove_passive_nodes(self, round : int) -> bool:
        # The node is considered passive after immidiately after the parameter resetting
        return round >= self.unlearning_round
    
    def is_active(self, round):
        return round == self.unlearning_round
    
    def before_training(self):
        # Reset the model parameters and learning rate to their initial values
        self.model.load_state_dict(self.initial_state_dict)
        self.model.modify_learning_rate(self.initial_learning_rate)
        # Set the model training step to avoid immidiate retraining after resetting
        self.previous_training_step = self.model.training_step
        self.model.training_step = self.model.training_step_zero_loss

    def after_training(self):
        # Reset the model training step to its previous value
        self.model.training_step = self.previous_training_step


class GradientAscent(UnlearningCycleExtension):
    """
    This extension implements the gradient ascent unlearning method.
    It modifies the model's training step to perform gradient ascent
    during the unlearning round, adjusts the gradient clipping value,
    and scales the model weight for update distribution.
    """
    def __init__(
            self,
            trainer : Lightning,
            is_unlearning_node : bool,
            unlearning_params : dict,
            ):
        self.model: NebulaModel = trainer.model
        self.datamodule: DataModule = trainer.datamodule
        self.trainer: Lightning = trainer
        self.is_unlearning_node: bool = is_unlearning_node
        self.unlearning_round: int = unlearning_params["unlearning_round"]
        self.gradient_clip_val: float = unlearning_params["gradient_clip_val"]
        self.weight_factor: int = unlearning_params["weight_factor"]
        self.previous_training_step = None
        self.previous_gradient_clip_val = None
        self.previous_weight = None

    def remove_passive_nodes(self, round : int) -> bool:
        # The node is considered passive after the gradient ascent update is distributed
        return round > self.unlearning_round
    
    def is_active(self, round):
        # The unlearning method is activated only in the unlearning round for the unlearning nodes
        return round == self.unlearning_round and self.is_unlearning_node

    def before_training(self):
        # Set the model training step to gradient ascent and remember the previous one
        self.previous_training_step = self.model.training_step
        self.model.training_step = self.model.training_step_gradient_ascent
        # Set the training gradient clip value and remember the previous one
        self.previous_gradient_clip_val = self.trainer.train_gradient_clip_val
        self.trainer.train_gradient_clip_val = self.gradient_clip_val

    def after_training(self):
        # Reset the model training step and gradient clip value to their previous values
        self.model.training_step = self.previous_training_step
        self.trainer.train_gradient_clip_val = self.previous_gradient_clip_val
    
    def before_publishing(self):
        # Set the model weight for update distribution and remember the previous weight
        self.previous_weight = self.datamodule.model_weight
        self.datamodule.model_weight *= self.weight_factor

    def after_publishing(self):
        # Reset the model weight to its previous value
        self.datamodule.model_weight = self.previous_weight