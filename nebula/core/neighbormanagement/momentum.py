import asyncio
import logging
from collections import deque
import os

from nebula.core.utils.locker import Locker
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from nebula.core.neighbormanagement.nodemanager import NodeManager

MAX_HISTORIC_SIZE = 10

class Momentum():

    def __init__(
        self,  
        node_manager : "NodeManager",
        nodes,
    ):
        self._node_manager = node_manager
        self._similarities_historic = {node_id: deque(maxlen=MAX_HISTORIC_SIZE) for node_id in nodes}
        self._similarities_historic_lock = Locker(name="__similarities_historic_lock", async_lock=True)

    @property
    def nm(self):
        return self._node_manager

    async def add_similarity_to_node(self, node_id, sim_value):
        logging.info(f"Adding | node ID: {node_id}, cossine similarity value: {sim_value}")
        self._similarities_historic_lock.acquire_async()
        self._similarities_historic[node_id].append(sim_value)
        self._similarities_historic_lock.release_async()

    async def update_node(self, node_id, remove=False):
        self._similarities_historic_lock.acquire_async()
        if remove:
            self._similarities_historic.pop(node_id, None)
        else:
            self._similarities_historic.update({node_id: deque(maxlen=MAX_HISTORIC_SIZE)})
        self._similarities_historic_lock.release_async()

    async def apply_similarity_weights(self, updates: dict):
        pass    