from abc import abstractmethod
from functools import wraps
import asyncio
import importlib
import logging
from copy import deepcopy
from typing import Any

import numpy as np
import torch
from torchmetrics.functional import pairwise_cosine_similarity

from nebula.addons.attacks.poisoning.modelpoison import modelPoison
from nebula.core.datasets.datamodule import DataModule

# To take into account:
# - Malicious nodes do not train on their own data
# - Malicious nodes aggregate the weights of the other nodes, but not their own
# - The received weights may be the node own weights (aggregated of neighbors), or
#   if the attack is performed specifically for one of the neighbors, it can take
#   its weights only (should be more effective if they are different).


def create_attack(attack_name, communications_manager, **kwargs):
    """
    Function to create an attack object from its name.
    """
    if attack_name == "GLLNeuronInversionAttack":
        return GLLNeuronInversionAttack()
    if attack_name == "NoiseInjectionAttack":
        valid_params = {k: v for k, v in kwargs.items() if k in ["strength"]}
        return NoiseInjectionAttack(**valid_params)
    elif attack_name == "SwappingWeightsAttack":
        valid_params = {k: v for k, v in kwargs.items() if k in ["layer_idx"]}
        return SwappingWeightsAttack(**valid_params)
    elif attack_name == "DelayerAttack":
        valid_params = {k: v for k, v in kwargs.items() if k in ["delay"]}
        return DelayerAttack(communications_manager, **valid_params)
    elif attack_name == "Label Flipping":
        valid_params = {k: v for k, v in kwargs.items() if k in ["poisoned_ratio" , "poisoned_percent", "targeted", "target_label", "target_changed_label", "noise_type"]}
        return LabelFlippingAttack(**valid_params)
    elif attack_name == "Sample Poisoning":
        valid_params = {k: v for k, v in kwargs.items() if k in ["poisoned_ratio" , "poisoned_percent", "targeted", "target_label", "target_changed_label", "noise_type"]}
        return SamplePoisoningAttack(**valid_params)
    elif attack_name == "Model Poisoning":
        valid_params = {k: v for k, v in kwargs.items() if k in ["poisoned_ratio", "noise_type"]}
        return ModelPoisonAttack(**valid_params)
    else:
        return None
    
async def create_malicious_behaviour(function_route: str, malicious_behaviour):
        """
        Replace dinamically a function to a malicious behaviour.
 
        Args:
            function_route (str): Route to class and method, format: 'module.class.method'.
            malicious_behaviour (callable): Malicious function that will replace the previous one.
 
        Returns:
            None
         """
        try:
            *module_route, function_name = function_route.rsplit(".", maxsplit=2)
            module = module_route[0]
            class_name = module_route[1]
            # class_name, function_name = class_and_func.split(".")
            logging.info(f"[FER] module = {module} class name = {class_name} function name = {function_name}")

            # Import module
            module_obj = importlib.import_module(module)

            # get class
            changing_class = getattr(module_obj, class_name)

            # Verify class got that function
            if not hasattr(changing_class, function_name):
                raise AttributeError(f"Class '{class_name}' got no method named: '{function_name}'.")

            # Replace old method to new function
            setattr(changing_class, function_name, malicious_behaviour)
            logging.info(f"Function '{function_name}' has been replaced with '{malicious_behaviour.__name__}'.")
        except Exception as e:
            logging.error(f"Error replacing function: {e}")


#######################
# Communication Attacks#
#######################


class CommunicationAttack:
    # async def __call__(self, *args: Any, **kwds: Any) -> Any:
    #     return await self.attack(*args, **kwds)

    @abstractmethod
    async def attack(self, received_weights):
        """
        Function to perform the attack on the received weights. It should return the
        attacked weights.
        """
        raise NotImplementedError


class DelayerAttack(CommunicationAttack):
    def __init__(self, communications_manager, delay):
        super().__init__()
        self.communications_manager = communications_manager
        self.delay = delay

    def delay_decorator(self, delay):
        """
        Decorator that adds a delay to the original method.
        """
        def decorator(func):
            @wraps(func)
            async def wrapper(self, *args, **kwargs):
                logging.info(f"[DelayerAttack] Adding delay of {delay} seconds")
                await asyncio.sleep(delay)
                return await func(self, *args, **kwargs)  # Call the original method
            return wrapper
        return decorator

    def inject_malicious_behaviour(self, target_class, method_name):
        """
        Inject the decorator into the target class's method.
        
        Args:
            target_class: Class where the method to be replaced is located.
            method_name: Name of the original method.
        """
        # Get the original method
        original_method = getattr(target_class, method_name)

        # Apply the decorator to the original method
        decorated_method = self.delay_decorator(self.delay)(original_method)

        # Replace the method in the class with the decorated method
        setattr(target_class, method_name, decorated_method)

    async def attack(self):
        logging.info("Creating [DelayerAttack]")
        self.inject_malicious_behaviour(self.communications_manager.__class__, "handle_model_message")

#################
# Dataset Attacks#
#################


class DatasetAttack:
    @abstractmethod
    def setMaliciousDataset(self, dataset):
        raise NotImplementedError


class LabelFlippingAttack(DatasetAttack):
    def __init__(self, poisoned_ratio, poisoned_percent, targeted, target_label, target_changed_label, noise_type):
        super().__init__()
        self.dataset = None
        self.poisoned_percent=poisoned_percent
        self.poisoned_ratio=poisoned_ratio
        self.targeted=targeted
        self.target_label=target_label
        self.target_changed_label=target_changed_label
        self.noise_type=noise_type

    def setMaliciousDataset(self, dataset):
        logging.info(f"[LabelFlippingAttack] Performing Label Flipping attack targeted {self.targeted}")
        self.dataset = DataModule(
            train_set=dataset.train_set,
            train_set_indices=dataset.train_set_indices,
            test_set=dataset.test_set,
            test_set_indices=dataset.test_set_indices,
            local_test_set_indices=dataset.local_test_set_indices,
            num_workers=dataset.num_workers,
            partition_id=dataset.partition_id,
            partitions_number=dataset.partitions_number,
            batch_size=dataset.batch_size,
            label_flipping=True,
            poisoned_percent=self.poisoned_percent,
            poisoned_ratio=self.poisoned_ratio,
            targeted=self.targeted,
            target_label=self.target_label,
            target_changed_label=self.target_changed_label,
            noise_type=self.noise_type,
        )
        return self.dataset


class SamplePoisoningAttack(DatasetAttack):
    def __init__(self, poisoned_ratio, poisoned_percent, targeted, target_label, target_changed_label, noise_type):
        super().__init__()
        self.dataset = None
        self.poisoned_percent=poisoned_percent
        self.poisoned_ratio=poisoned_ratio
        self.targeted=targeted
        self.target_label=target_label
        self.target_changed_label=target_changed_label
        self.noise_type=noise_type

    def setMaliciousDataset(self, dataset):
        logging.info("[SamplePoisoningAttack] Performing Sample Poisoning attack")
        self.dataset = DataModule(
            train_set=dataset.train_set,
            train_set_indices=dataset.train_set_indices,
            test_set=dataset.test_set,
            test_set_indices=dataset.test_set_indices,
            local_test_set_indices=dataset.local_test_set_indices,
            num_workers=dataset.num_workers,
            partition_id=dataset.partition_id,
            partitions_number=dataset.partitions_number,
            batch_size=dataset.batch_size,
            data_poisoning=True,
            poisoned_percent=self.poisoned_percent,
            poisoned_ratio=self.poisoned_ratio,
            targeted=self.targeted,
            target_label=self.target_label,
            target_changed_label=self.target_changed_label,
            noise_type=self.noise_type,
        )
        return self.dataset


###############
# Model Attacks#
###############


class ModelAttack:
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.ModelAttack(*args, **kwds)
    
    @abstractmethod
    def ModelAttack(self, received_weights):
        raise NotImplementedError


class ModelPoisonAttack(ModelAttack):
    def __init__(self, poisoned_ratio, noise_type):
        super().__init__()
        self.model = None
        self.poisoned_ratio = poisoned_ratio
        self.noise_type = noise_type

    def ModelAttack(self, received_weights):
        logging.info("[ModelPoisonAttack] Performing model poison attack")
        received_weights = modelPoison(received_weights, self.poisoned_ratio, self.noise_type)
        return received_weights

 
class GLLNeuronInversionAttack(ModelAttack):
    """
    Function to perform neuron inversion attack on the received weights.
    """

    def __init__(self):
        super().__init__()

    def ModelAttack(self, received_weights):
        logging.info("[GLLNeuronInversionAttack] Performing neuron inversion attack")
        lkeys = list(received_weights.keys())
        logging.info(f"Layer inverted: {lkeys[-2]}")
        received_weights[lkeys[-2]].data = torch.rand(received_weights[lkeys[-2]].shape) * 10000
        return received_weights


class NoiseInjectionAttack(ModelAttack):
    """
    Function to perform noise injection attack on the received weights.
    """

    def __init__(self, strength=10000):
        super().__init__()
        self.strength = strength

    def ModelAttack(self, received_weights):
        logging.info(f"[NoiseInjectionAttack] Performing noise injection attack with a strength of {self.strength}")
        lkeys = list(received_weights.keys())
        for k in lkeys:
            logging.info(f"Layer noised: {k}")
            received_weights[k].data += torch.randn(received_weights[k].shape) * self.strength
        return received_weights


class SwappingWeightsAttack(ModelAttack):
    """
    Function to perform swapping weights attack on the received weights. Note that this
    attack performance is not consistent due to its stochasticity.

    Warning: depending on the layer the code may not work (due to reshaping in between),
    or it may be slow (scales quadratically with the layer size).
    Do not apply to last layer, as it would make the attack detectable (high loss
    on malicious node).
    """

    def __init__(self, layer_idx=0):
        super().__init__()
        self.layer_idx = layer_idx

    def ModelAttack(self, received_weights):
        logging.info("[SwappingWeightsAttack] Performing swapping weights attack")
        lkeys = list(received_weights.keys())
        wm = received_weights[lkeys[self.layer_idx]]

        # Compute similarity matrix
        sm = torch.zeros((wm.shape[0], wm.shape[0]))
        for j in range(wm.shape[0]):
            sm[j] = pairwise_cosine_similarity(wm[j].reshape(1, -1), wm.reshape(wm.shape[0], -1))

        # Check rows/cols where greedy approach is optimal
        nsort = np.full(sm.shape[0], -1)
        rows = []
        for j in range(sm.shape[0]):
            k = torch.argmin(sm[j])
            if torch.argmin(sm[:, k]) == j:
                nsort[j] = k
                rows.append(j)
        not_rows = np.array([i for i in range(sm.shape[0]) if i not in rows])

        # Ensure the rest of the rows are fully permuted (not optimal, but good enough)
        nrs = deepcopy(not_rows)
        nrs = np.random.permutation(nrs)
        while np.any(nrs == not_rows):
            nrs = np.random.permutation(nrs)
        nsort[not_rows] = nrs
        nsort = torch.tensor(nsort)

        # Apply permutation to weights
        received_weights[lkeys[self.layer_idx]] = received_weights[lkeys[self.layer_idx]][nsort]
        received_weights[lkeys[self.layer_idx + 1]] = received_weights[lkeys[self.layer_idx + 1]][nsort]
        if self.layer_idx + 2 < len(lkeys):
            received_weights[lkeys[self.layer_idx + 2]] = received_weights[lkeys[self.layer_idx + 2]][:, nsort]
        return received_weights
