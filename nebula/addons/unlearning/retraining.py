import logging
from copy import deepcopy

from nebula.core.models.nebulamodel import NebulaModel
from nebula.addons.unlearning.extension import LearningCycleExtension

class RetrainingCycleExtension(LearningCycleExtension):
    """
    Base class for retraining cycle extensions. This class should
    do nothing when set as active extension and serve as a default.
    """
    def is_setup_round(self, round: int) -> bool:
        return False
    
    def setup(self):
        pass

class KnowledgeDistillation(RetrainingCycleExtension):
    """
    This extension implements knowledge distillation during the retraining phase.
    It uses a teacher model to guide the student model (the current model)
    towards better performance after unlearning. It sets the alpha and temperature
    parameters for the knowledge distillation process. 
    """
    def __init__(
            self,
            model: NebulaModel,
            unlearning_params: dict,
            ):
        
        self.model = model
        self.teacher = None
        self.unlearning_round = unlearning_params["unlearning_round"]
        self.retraining_rounds = unlearning_params["retraining_rounds"]
        self.alpha = unlearning_params["alpha"]
        self.temperature = unlearning_params["temperature"]
        self.previous_training_step = None

    def is_setup_round(self, round: int) -> bool:
        # Initialize the teacher model before the unlearning is happening
        return round == self.unlearning_round

    def setup(self):
        # Initialize the teacher model as a deep copy of the current model.
        self.teacher = deepcopy(self.model)
        logging.info(f"Teacher model for Knowledge Distillation is set up.")
    
    def is_active(self, round):
        # The knowledge distillation is activated during the retraining phase
        # after the unlearning round and for the specified number of retraining rounds.
        return self.unlearning_round < round <= self.unlearning_round + self.retraining_rounds

    def before_training(self):
        # Set the alpha and temperature parameters for knowledge distillation
        # and set the teacher model as submodule for the student model to use during
        # training. Finally, set the training step to the knowledge distillation step
        # and remember the previous training step
        self.model.alpha = self.alpha
        self.model.temperature = self.temperature
        self.model.teacher = deepcopy(self.teacher)
        self.previous_training_step = self.model.training_step
        self.model.training_step = self.model.training_step_knowledge_distillation
        logging.info(f"Alpha: {self.alpha}, Temperature: {self.temperature}")

    def after_training(self):
        # Reset the model training step to its previous value and delete the teacher
        # model such that it does not count as a submodule of the student model
        self.model.training_step = self.previous_training_step
        del self.model.alpha
        del self.model.temperature
        del self.model.teacher
        logging.info("Resetting Knowledge Distillation parameters of student model.")