# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: nebula/core/pb/nebula.proto
# Protobuf Python Version: 5.28.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
_runtime_version.ValidateProtobufRuntimeVersion(
    _runtime_version.Domain.PUBLIC,
    5,
    28,
    1,
    '',
    'nebula/core/pb/nebula.proto'
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x1bnebula/core/pb/nebula.proto\x12\x06nebula\"\xe4\x02\n\x07Wrapper\x12\x0e\n\x06source\x18\x01 \x01(\t\x12\x35\n\x11\x64iscovery_message\x18\x02 \x01(\x0b\x32\x18.nebula.DiscoveryMessageH\x00\x12\x31\n\x0f\x63ontrol_message\x18\x03 \x01(\x0b\x32\x16.nebula.ControlMessageH\x00\x12\x37\n\x12\x66\x65\x64\x65ration_message\x18\x04 \x01(\x0b\x32\x19.nebula.FederationMessageH\x00\x12-\n\rmodel_message\x18\x05 \x01(\x0b\x32\x14.nebula.ModelMessageH\x00\x12\x37\n\x12\x63onnection_message\x18\x06 \x01(\x0b\x32\x19.nebula.ConnectionMessageH\x00\x12\x33\n\x10response_message\x18\x07 \x01(\x0b\x32\x17.nebula.ResponseMessageH\x00\x42\t\n\x07message\"\x9e\x01\n\x10\x44iscoveryMessage\x12/\n\x06\x61\x63tion\x18\x01 \x01(\x0e\x32\x1f.nebula.DiscoveryMessage.Action\x12\x10\n\x08latitude\x18\x02 \x01(\x02\x12\x11\n\tlongitude\x18\x03 \x01(\x02\"4\n\x06\x41\x63tion\x12\x0c\n\x08\x44ISCOVER\x10\x00\x12\x0c\n\x08REGISTER\x10\x01\x12\x0e\n\nDEREGISTER\x10\x02\"\x9a\x01\n\x0e\x43ontrolMessage\x12-\n\x06\x61\x63tion\x18\x01 \x01(\x0e\x32\x1d.nebula.ControlMessage.Action\x12\x0b\n\x03log\x18\x02 \x01(\t\"L\n\x06\x41\x63tion\x12\t\n\x05\x41LIVE\x10\x00\x12\x0c\n\x08OVERHEAD\x10\x01\x12\x0c\n\x08MOBILITY\x10\x02\x12\x0c\n\x08RECOVERY\x10\x03\x12\r\n\tWEAK_LINK\x10\x04\"\xcd\x01\n\x11\x46\x65\x64\x65rationMessage\x12\x30\n\x06\x61\x63tion\x18\x01 \x01(\x0e\x32 .nebula.FederationMessage.Action\x12\x11\n\targuments\x18\x02 \x03(\t\x12\r\n\x05round\x18\x03 \x01(\x05\"d\n\x06\x41\x63tion\x12\x14\n\x10\x46\x45\x44\x45RATION_START\x10\x00\x12\x0e\n\nREPUTATION\x10\x01\x12\x1e\n\x1a\x46\x45\x44\x45RATION_MODELS_INCLUDED\x10\x02\x12\x14\n\x10\x46\x45\x44\x45RATION_READY\x10\x03\"A\n\x0cModelMessage\x12\x12\n\nparameters\x18\x01 \x01(\x0c\x12\x0e\n\x06weight\x18\x02 \x01(\x03\x12\r\n\x05round\x18\x03 \x01(\x05\"l\n\x11\x43onnectionMessage\x12\x30\n\x06\x61\x63tion\x18\x01 \x01(\x0e\x32 .nebula.ConnectionMessage.Action\"%\n\x06\x41\x63tion\x12\x0b\n\x07\x43ONNECT\x10\x00\x12\x0e\n\nDISCONNECT\x10\x01\"#\n\x0fResponseMessage\x12\x10\n\x08response\x18\x01 \x01(\t\"B\n\x11ReputationMessage\x12\x0f\n\x07node_id\x18\x01 \x01(\t\x12\r\n\x05score\x18\x02 \x01(\x02\x12\r\n\x05round\x18\x03 \x01(\x05\"c\n\x12\x46loodAttackMessage\x12\x13\n\x0b\x61ttacker_id\x18\x01 \x01(\t\x12\x11\n\tfrequency\x18\x02 \x01(\x05\x12\x10\n\x08\x64uration\x18\x03 \x01(\x05\x12\x13\n\x0btarget_node\x18\x04 \x01(\tb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'nebula.core.pb.nebula_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_WRAPPER']._serialized_start=40
  _globals['_WRAPPER']._serialized_end=396
  _globals['_DISCOVERYMESSAGE']._serialized_start=399
  _globals['_DISCOVERYMESSAGE']._serialized_end=557
  _globals['_DISCOVERYMESSAGE_ACTION']._serialized_start=505
  _globals['_DISCOVERYMESSAGE_ACTION']._serialized_end=557
  _globals['_CONTROLMESSAGE']._serialized_start=560
  _globals['_CONTROLMESSAGE']._serialized_end=714
  _globals['_CONTROLMESSAGE_ACTION']._serialized_start=638
  _globals['_CONTROLMESSAGE_ACTION']._serialized_end=714
  _globals['_FEDERATIONMESSAGE']._serialized_start=717
  _globals['_FEDERATIONMESSAGE']._serialized_end=922
  _globals['_FEDERATIONMESSAGE_ACTION']._serialized_start=822
  _globals['_FEDERATIONMESSAGE_ACTION']._serialized_end=922
  _globals['_MODELMESSAGE']._serialized_start=924
  _globals['_MODELMESSAGE']._serialized_end=989
  _globals['_CONNECTIONMESSAGE']._serialized_start=991
  _globals['_CONNECTIONMESSAGE']._serialized_end=1099
  _globals['_CONNECTIONMESSAGE_ACTION']._serialized_start=1062
  _globals['_CONNECTIONMESSAGE_ACTION']._serialized_end=1099
  _globals['_RESPONSEMESSAGE']._serialized_start=1101
  _globals['_RESPONSEMESSAGE']._serialized_end=1136
  _globals['_REPUTATIONMESSAGE']._serialized_start=1138
  _globals['_REPUTATIONMESSAGE']._serialized_end=1204
  _globals['_FLOODATTACKMESSAGE']._serialized_start=1206
  _globals['_FLOODATTACKMESSAGE']._serialized_end=1305
# @@protoc_insertion_point(module_scope)
