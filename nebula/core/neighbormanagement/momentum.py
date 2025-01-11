import asyncio
import logging
from collections import deque
from nebula.core.utils.helper import cosine_metric
from nebula.core.utils.locker import Locker
import numpy as np

from typing import TYPE_CHECKING, Callable, OrderedDict, Optional
if TYPE_CHECKING:
    from nebula.core.neighbormanagement.nodemanager import NodeManager

SimilarityMetricType = Callable[[OrderedDict, OrderedDict, bool], Optional[float]]

MAX_HISTORIC_SIZE = 10      # Number of historic data storaged
GLOBAL_PRIORITY = 0.5       # Parameter to priorize global vs local metrics
K = 1                       # Sigmoid smoother factor

class Momentum():

    def __init__(
        self,  
        node_manager: "NodeManager",
        nodes,
        variance=True,
        similarity_metric : SimilarityMetricType = cosine_metric,
        global_priority=GLOBAL_PRIORITY,
    ):
        self._node_manager = node_manager
        self._similarities_historic = {node_id: deque(maxlen=MAX_HISTORIC_SIZE) for node_id in nodes}
        self._similarities_historic_lock = Locker(name="_similarities_historic_lock", async_lock=True)
        self._model_similarity_metric_lock = Locker(name="_model_similarity_metric_lock", async_lock=True)
        self._model_similarity_metric = similarity_metric
        self._global_prio = global_priority
        self._variance_status = variance

    @property
    def nm(self):
        return self._node_manager
    
    @property
    def msm(self):
        return self._model_similarity_metric

    async def _add_similarity_to_node(self, node_id, sim_value):
        logging.info(f"Adding | node ID: {node_id}, cossine similarity value: {sim_value}")
        self._similarities_historic_lock.acquire_async()
        self._similarities_historic[node_id].append(sim_value)
        self._similarities_historic_lock.release_async()
        
    async def _get_similarity_historic(self, addrs):
        """
            Get historic storaged for node IDs on 'addrs'

        Args:
            addrs (List)): List of node IDs that has sent update this round
        """
        self._similarities_historic_lock.acquire_async()
        historic = {}
        for key, value in self._similarities_historic.items():
            if key in addrs:
                historic[key] = value
        self._similarities_historic_lock.release_async()
        return historic    

    async def update_node(self, node_id, remove=False):
        self._similarities_historic_lock.acquire_async()
        if remove:
            self._similarities_historic.pop(node_id, None)
        else:
            self._similarities_historic.update({node_id: deque(maxlen=MAX_HISTORIC_SIZE)})
        self._similarities_historic_lock.release_async()
        
    async def change_similarity_metric(self, new_metric: SimilarityMetricType):
        self._model_similarity_metric_lock.acquire_async()
        self.msm = new_metric
        # maybe we should remove historic data due to incongruous data
        self._model_similarity_metric_lock.release_async()
        
    async def _calculate_similarities(self, updates: dict):
        """
            Function to calculate similarity between local model and models received
            using metric function. The value is storaged on the historic

        Args:
            updates (dict): {node ID: model}
        """
        logging.info(f"Calculating | Model Similarity values are being calculated...")
        model = self.nm.engine.trainer.get_model_parameters()
        for addr,update in updates.items():
            cosine_value = self._model_similarity_metric(
                model,
                update,
                similarity=True,
            )
            await self._add_similarity_to_node(addr, cosine_value)    

    async def calculate_similarity_weights(self, updates: dict):
        logging.info("Calculating | Momemtum weights are being calculated...")
        self._model_similarity_metric_lock.acquire_async()
        await self._calculate_similarities(updates)
        historic = await self._get_similarity_historic(updates.keys())
        similarities = [node_sim[-1] for node_sim in historic.values() if node_sim]
        self._model_similarity_metric_lock.release_async() 