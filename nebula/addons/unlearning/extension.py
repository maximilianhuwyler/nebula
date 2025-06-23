import logging
from enum import Enum, auto

from nebula.config.config import Config
from nebula.core.training.lightning import Lightning
from nebula.core.network.communications import CommunicationsManager

class NodeState(Enum):
    ACTIVE = auto()
    PASSIVE = auto()

class LearningCycleExtensionManager:
    """
    This class manages the learning cycle extensions for unlearning and retraining.
    It initializes the extensions based on the configuration and manages the state of the node
    during the learning cycle. It also provides methods to configure the expected nodes and
    handle the unlearning and retraining processes.
    """
    def __init__(
            self,
            trainer : Lightning,
            cm : CommunicationsManager,
            config=Config,
            ):
        self.cm = cm
        self.config = config
        self.idx = self.config.participant["device_args"]["idx"]
        self.addr = self.config.participant["network_args"]["addr"]
        self.unlearning_args = self.config.participant["unlearning_args"]

        # Initialize default values
        self.is_unlearning_node = False
        from nebula.addons.unlearning.unlearning import UnlearningCycleExtension
        self.unlearning_extension = UnlearningCycleExtension()
        from nebula.addons.unlearning.retraining import RetrainingCycleExtension
        self.retraining_extension = RetrainingCycleExtension()

        # Configure unlearning and retraining
        self.unlearning_method_name = self.unlearning_args["unlearning_method"]
        if self.unlearning_method_name != "No Unlearning":
            self.unlearning_params = self.unlearning_args["unlearning_params"]
            self.unlearning_nodes = [int(node) for node in self.unlearning_params["unlearning_nodes"]]
            self.is_unlearning_node = self.idx in self.unlearning_nodes

            if self.unlearning_method_name == "Parameter Resetting":
                from nebula.addons.unlearning.unlearning import ParameterResetting
                self.unlearning_extension = ParameterResetting(trainer.model,
                                                               self.unlearning_params)
            if self.unlearning_method_name == "Gradient Ascent":
                from nebula.addons.unlearning.unlearning import GradientAscent
                self.unlearning_extension = GradientAscent(trainer,
                                                           self.is_unlearning_node,
                                                           self.unlearning_params)

            self.retraining_method_name = self.unlearning_args["unlearning_params"]["retraining_method"]
            if self.retraining_method_name != "No Retraining":
                if self.retraining_method_name == "Knowledge Distillation":
                    from nebula.addons.unlearning.retraining import KnowledgeDistillation
                    self.retraining_extension = KnowledgeDistillation(trainer.model, 
                                                                      self.unlearning_params)

    def remove_passive_nodes(
            self,
            expected_nodes: set,
            ):
        """
        Removes the nodes from the expected nodes set after the unlearning is done and the node
        is in a passive state.

        Parameters:
            expected_nodes: set: The set of ids of nodes that are expected to provide updates
            in the round.
        """
        if self.is_unlearning_node:
            logging.info(f"Removing itself from expected nodes")
            expected_nodes.discard(self.addr)
        for addr, conn in self.cm.connections.items():
            node_id = int(getattr(conn, "id", None))
            if node_id in self.unlearning_nodes:
                expected_nodes.discard(addr)
                logging.info(f"Removing unlearning node {node_id} with address {addr} from expected nodes")

    def configure_round(
            self,
            round: int,
            expected_nodes: set
            ) -> NodeState:
        """
        Configures the expected nodes, unlearning and retraining extensions for the current round.

        Parameters:
            round: int: The current round of the training.
            expected_nodes: set: The set of ids of nodes that are expected to provide updates
            in the round.
        
        Returns:
            NodeState: The state of the node in this round. Either ACTIVE, which means the node
            provides updates, or PASSIVE, which means the node does not provide updates.
        """
        # Depending on unlearning method the unlearning nodes should be removed from the expected nodes
        if self.unlearning_extension.remove_passive_nodes(round):
            self.remove_passive_nodes(expected_nodes)

        # Initialize the retraining extension if needed
        if self.retraining_extension.do_init(round):
            self.retraining_extension.init()

        # Set the active extension based on the current round
        if self.unlearning_extension.is_activate(round):
            self.active_extension = self.unlearning_extension
        elif self.retraining_extension.is_activate(round):
            self.active_extension = self.retraining_extension
        else:
            # If no unlearning or retraining is active, use the default learning cycle extension
            self.active_extension = LearningCycleExtension()

        # Determine the node state based on whether an update from the node is expected
        return NodeState.ACTIVE if self.addr in expected_nodes else NodeState.PASSIVE
        
    # Wrapper methods for the active extension
    def before_training(self):
        self.active_extension.before_training()

    def after_training(self):
        self.active_extension.after_training()

    def before_publishing(self):
        self.active_extension.before_publishing()

    def after_publishing(self):
        self.active_extension.after_publishing()


class LearningCycleExtension:
    def is_activate(self, round: int) -> bool:
        """
        Checks if the extension is activated for the current round.
        
        Parameters:
            round: int: The current round of the training.

        Returns:
            bool: True if the extension is activated for the current round, False otherwise.
        """
        return False

    def before_training(self):
        """
        Routine to be executed before the training step.
        """
        pass

    def after_training(self):
        """
        Routine to be executed after the training step.
        """
        pass

    def before_publishing(self):
        """
        Routine to be executed before publishing the model updates.
        """
        pass

    def after_publishing(self):
        """
        Routine to be executed after publishing the model updates.
        """
        pass