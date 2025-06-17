import logging
from enum import Enum, auto

from nebula.config.config import Config
from nebula.core.models.nebulamodel import NebulaModel
from nebula.core.datasets.datamodule import DataModule
from nebula.core.network.communications import CommunicationsManager
from nebula.core.unlearning.unlearning import ParameterResetting, GradientAscent
from nebula.core.unlearning.retraining import KnowledgeDistillation

class ExtendedState(Enum):
    NORMAL = auto()
    UNLEARNING = auto()
    FINISHED = auto()

class UnlearningManager:
    """
    Responsible for removing the influence of specific node from a DFL federation.
    """

    def __init__(
            self,
            model : NebulaModel,
            datamodule : DataModule,
            cm : CommunicationsManager,
            config=Config,
            ):
        """
        Initialize the Unlearner.

        Args:
            model: The model to unlearn from.
            data: The data or indices to unlearn.
            config: Optional configuration parameters.
        """
        self.config = config
        self.cm = cm
        self.idx = config.participant["device_args"]["idx"]
        self.addr = config.participant["network_args"]["addr"]
        self.unlearning_args = self.config.participant["unlearning_args"]
        self.unlearning_round = self.config.participant["unlearning_args"]["unlearning_round"]
        self.unlearning_nodes = self.config.participant["unlearning_args"]["unlearning_nodes"]
        self.is_unlearning_node = self.idx in self.unlearning_nodes

        self.unlearning_method_name = self.config.participant["unlearning_args"]["unlearning_method"]
        self.unlearning_rounds = self.config.participant["unlearning_args"]["unlearning_rounds"]
        self.unlearning_rounds_end = self.unlearning_round + self.unlearning_rounds - 1

        self.retraining_method_name = self.config.participant["unlearning_args"]["retraining_method"]
        self.retraining_rounds = self.config.participant["unlearning_args"]["retraining_rounds"]
        self.retraining_rounds_end = self.unlearning_rounds_end + self.retraining_rounds


        self.unlearning_method = None
        if self.unlearning_method_name == "Parameter Resetting":
            self.unlearning_method = ParameterResetting(model)
        elif self.unlearning_method_name == "Gradient Ascent":
            self.unlearning_method = GradientAscent(model, datamodule)

        self.retraining_method = None
        if self.retraining_method_name == "Knowledge Distillation":
            self.retraining_method = KnowledgeDistillation(model)

    def remove_unlearning_nodes(
            self,
            expected_nodes : set,
            ):
        if self.is_unlearning_node:
            logging.info(f"Removing itself from expected nodes")
            expected_nodes.discard(self.addr)
        for addr, conn in self.cm.connections.items():
            node_id = getattr(conn, "id", None)
            if node_id in self.unlearning_nodes:
                expected_nodes.discard(addr)
                logging.info(f"Removing unlearning node {node_id} with address {addr} from expected nodes")

    def configure_round(
            self,
            round : int,
            expected_nodes : set
            ) -> ExtendedState:

        if round < self.unlearning_round:
            return ExtendedState.NORMAL
        
        if self.unlearning_method.remove_unlearning_nodes:
            self.remove_unlearning_nodes(round, expected_nodes)
        
        if self.unlearning_round <= round <= self.unlearning_rounds_end:
            self.active_mechanism = self.unlearning_method
        elif self.unlearning_rounds_end < round <= self.retraining_rounds_end:
            self.active_mechanism = self.retraining_method
        
        if self.is_unlearning_node and round <= self.unlearning_rounds_end:
            return ExtendedState.UNLEARNING
        elif self.is_unlearning_node:
            return ExtendedState.FINISHED
        elif round <= self.retraining_rounds_end:
            return ExtendedState.UNLEARNING
        else:
            return ExtendedState.NORMAL


        
    def before_training(self):
        self.active_mechanism.before_training()

    def after_training(self):
        self.active_mechanism.after_training()

    def before_publishing(self):
        self.active_mechanism.before_publishing()

    def after_publishing(self):
        self.active_mechanism.after_publishing()


class UnlearningMechanism:
    def before_training(self):
        pass

    def after_training(self):
        pass

    def before_publishing(self):
        pass

    def after_publishing(self):
        pass