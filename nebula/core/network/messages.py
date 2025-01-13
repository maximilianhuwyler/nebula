import logging
from typing import TYPE_CHECKING

from nebula.core.pb import nebula_pb2

if TYPE_CHECKING:
    from nebula.core.network.communications import CommunicationsManager


class MessagesManager:
    def __init__(self, addr, config, cm: "CommunicationsManager"):
        self.addr = addr
        self.config = config
        self.cm = cm

    def generate_discovery_message(self, action, latitude=0.0, longitude=0.0):
        message = nebula_pb2.DiscoveryMessage(
            action=action,
            latitude=latitude,
            longitude=longitude,
        )
        message_wrapper = nebula_pb2.Wrapper()
        message_wrapper.source = self.addr
        message_wrapper.discovery_message.CopyFrom(message)
        data = message_wrapper.SerializeToString()
        return data

    def generate_control_message(self, action, log="Control message"):
        message = nebula_pb2.ControlMessage(
            action=action,
            log=log,
        )
        message_wrapper = nebula_pb2.Wrapper()
        message_wrapper.source = self.addr
        message_wrapper.control_message.CopyFrom(message)
        data = message_wrapper.SerializeToString()
        return data

    def generate_federation_message(self, action, arguments=[], round=None):
        logging.info(f"Building federation message with [Action {action}], arguments {arguments}, and round {round}")

        message = nebula_pb2.FederationMessage(
            action=action,
            arguments=[str(arg) for arg in (arguments or [])],
            round=round,
        )
        message_wrapper = nebula_pb2.Wrapper()
        message_wrapper.source = self.addr
        message_wrapper.federation_message.CopyFrom(message)
        data = message_wrapper.SerializeToString()
        return data

    def generate_model_message(self, round, serialized_model, weight=1):
        message = nebula_pb2.ModelMessage(
            round=round,
            parameters=serialized_model,
            weight=weight,
        )
        message_wrapper = nebula_pb2.Wrapper()
        message_wrapper.source = self.addr
        message_wrapper.model_message.CopyFrom(message)
        data = message_wrapper.SerializeToString()
        return data

    def generate_connection_message(self, action):
        message = nebula_pb2.ConnectionMessage(
            action=action,
        )
        message_wrapper = nebula_pb2.Wrapper()
        message_wrapper.source = self.addr
        message_wrapper.connection_message.CopyFrom(message)
        data = message_wrapper.SerializeToString()
        return data

    def generate_reputation_message(self, reputation):
        message = nebula_pb2.ReputationMessage(
            reputation=reputation,
        )
        message_wrapper = nebula_pb2.Wrapper()
        message_wrapper.source = self.addr
        message_wrapper.reputation_message.CopyFrom(message)
        data = message_wrapper.SerializeToString()
        return data

    def generate_nss_features_message(self, nss_features):
        message = nebula_pb2.NSSFeaturesMessage(
            cpu_percent=nss_features["cpu_percent"],
            bytes_sent=nss_features["bytes_sent"],
            bytes_received=nss_features["bytes_received"],
            loss=nss_features["loss"],
            data_size=nss_features["data_size"],
            sustainability=nss_features["sustainability"],
        )
        message_wrapper = nebula_pb2.Wrapper()
        message_wrapper.source = self.addr
        message_wrapper.nss_features_message.CopyFrom(message)
        data = message_wrapper.SerializeToString()
        return data

    def generate_vote_message(self):
        message = nebula_pb2.VoteMessage(
            vote=0,
        )
        message_wrapper = nebula_pb2.Wrapper()
        message_wrapper.source = self.addr
        message_wrapper.vote_message.CopyFrom(message)
        data = message_wrapper.SerializeToString()
        return data
    
    def generate_embedding_message(self, embedding, embedding_covariance):
        message = nebula_pb2.EmbeddingMessage(
            embedding=embedding,
            embedding_covariance=embedding_covariance,
        )
        message_wrapper = nebula_pb2.Wrapper()
        message_wrapper.source = self.addr
        message_wrapper.embedding_message.CopyFrom(message)
        data = message_wrapper.SerializeToString()
        return data
