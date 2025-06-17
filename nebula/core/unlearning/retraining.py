from copy import deepcopy
from types import MethodType
from typing import override

from nebula.core.models.nebulamodel import NebulaModel
from nebula.core.unlearning.unlearningmanager import UnlearningMechanism

class KnowledgeDistillation(UnlearningMechanism):

    def __init__(
            self,
            model : NebulaModel,
            ):
        
        self.model = model
        self.teacher = deepcopy(model)
        self.original_step = model.step

    @override
    def before_training(self):
        self.model.kd_alpha = self.alpha
        self.model.kd_temperature = self.temperature
        self.model.kd_teacher = deepcopy(self.teacher)
        self.model.step = MethodType(self.model.step_knowledge_distillation, self.model)

    @override
    def after_training(self):
        self.model.step = self.original_step
        del self.model.kd_alpha
        del self.model.kd_temperature
        del self.model.kd_teacher