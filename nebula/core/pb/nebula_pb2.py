# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: nebula.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0cnebula.proto\x12\x06nebula\"\xcd\x03\n\x07Wrapper\x12\x0e\n\x06source\x18\x01 \x01(\t\x12\x35\n\x11\x64iscovery_message\x18\x02 \x01(\x0b\x32\x18.nebula.DiscoveryMessageH\x00\x12\x31\n\x0f\x63ontrol_message\x18\x03 \x01(\x0b\x32\x16.nebula.ControlMessageH\x00\x12\x37\n\x12\x66\x65\x64\x65ration_message\x18\x04 \x01(\x0b\x32\x19.nebula.FederationMessageH\x00\x12-\n\rmodel_message\x18\x05 \x01(\x0b\x32\x14.nebula.ModelMessageH\x00\x12\x37\n\x12\x63onnection_message\x18\x06 \x01(\x0b\x32\x19.nebula.ConnectionMessageH\x00\x12\x33\n\x10response_message\x18\x07 \x01(\x0b\x32\x17.nebula.ResponseMessageH\x00\x12:\n\x14nss_features_message\x18\x08 \x01(\x0b\x32\x1a.nebula.NSSFeaturesMessageH\x00\x12+\n\x0cvote_message\x18\t \x01(\x0b\x32\x13.nebula.VoteMessageH\x00\x42\t\n\x07message\"\x9e\x01\n\x10\x44iscoveryMessage\x12/\n\x06\x61\x63tion\x18\x01 \x01(\x0e\x32\x1f.nebula.DiscoveryMessage.Action\x12\x10\n\x08latitude\x18\x02 \x01(\x02\x12\x11\n\tlongitude\x18\x03 \x01(\x02\"4\n\x06\x41\x63tion\x12\x0c\n\x08\x44ISCOVER\x10\x00\x12\x0c\n\x08REGISTER\x10\x01\x12\x0e\n\nDEREGISTER\x10\x02\"\x9a\x01\n\x0e\x43ontrolMessage\x12-\n\x06\x61\x63tion\x18\x01 \x01(\x0e\x32\x1d.nebula.ControlMessage.Action\x12\x0b\n\x03log\x18\x02 \x01(\t\"L\n\x06\x41\x63tion\x12\t\n\x05\x41LIVE\x10\x00\x12\x0c\n\x08OVERHEAD\x10\x01\x12\x0c\n\x08MOBILITY\x10\x02\x12\x0c\n\x08RECOVERY\x10\x03\x12\r\n\tWEAK_LINK\x10\x04\"\xcd\x01\n\x11\x46\x65\x64\x65rationMessage\x12\x30\n\x06\x61\x63tion\x18\x01 \x01(\x0e\x32 .nebula.FederationMessage.Action\x12\x11\n\targuments\x18\x02 \x03(\t\x12\r\n\x05round\x18\x03 \x01(\x05\"d\n\x06\x41\x63tion\x12\x14\n\x10\x46\x45\x44\x45RATION_START\x10\x00\x12\x0e\n\nREPUTATION\x10\x01\x12\x1e\n\x1a\x46\x45\x44\x45RATION_MODELS_INCLUDED\x10\x02\x12\x14\n\x10\x46\x45\x44\x45RATION_READY\x10\x03\"A\n\x0cModelMessage\x12\x12\n\nparameters\x18\x01 \x01(\x0c\x12\x0e\n\x06weight\x18\x02 \x01(\x03\x12\r\n\x05round\x18\x03 \x01(\x05\"l\n\x11\x43onnectionMessage\x12\x30\n\x06\x61\x63tion\x18\x01 \x01(\x0e\x32 .nebula.ConnectionMessage.Action\"%\n\x06\x41\x63tion\x12\x0b\n\x07\x43ONNECT\x10\x00\x12\x0e\n\nDISCONNECT\x10\x01\"#\n\x0fResponseMessage\x12\x10\n\x08response\x18\x01 \x01(\t\"\x8e\x01\n\x12NSSFeaturesMessage\x12\x13\n\x0b\x63pu_percent\x18\x01 \x01(\x02\x12\x12\n\nbytes_sent\x18\x02 \x01(\x03\x12\x16\n\x0e\x62ytes_received\x18\x03 \x01(\x03\x12\x0c\n\x04loss\x18\x04 \x01(\x02\x12\x11\n\tdata_size\x18\x05 \x01(\x03\x12\x16\n\x0esustainability\x18\x06 \x01(\x02\"\x1b\n\x0bVoteMessage\x12\x0c\n\x04vote\x18\x01 \x01(\x05\x62\x06proto3')



_WRAPPER = DESCRIPTOR.message_types_by_name['Wrapper']
_DISCOVERYMESSAGE = DESCRIPTOR.message_types_by_name['DiscoveryMessage']
_CONTROLMESSAGE = DESCRIPTOR.message_types_by_name['ControlMessage']
_FEDERATIONMESSAGE = DESCRIPTOR.message_types_by_name['FederationMessage']
_MODELMESSAGE = DESCRIPTOR.message_types_by_name['ModelMessage']
_CONNECTIONMESSAGE = DESCRIPTOR.message_types_by_name['ConnectionMessage']
_RESPONSEMESSAGE = DESCRIPTOR.message_types_by_name['ResponseMessage']
_NSSFEATURESMESSAGE = DESCRIPTOR.message_types_by_name['NSSFeaturesMessage']
_VOTEMESSAGE = DESCRIPTOR.message_types_by_name['VoteMessage']
_DISCOVERYMESSAGE_ACTION = _DISCOVERYMESSAGE.enum_types_by_name['Action']
_CONTROLMESSAGE_ACTION = _CONTROLMESSAGE.enum_types_by_name['Action']
_FEDERATIONMESSAGE_ACTION = _FEDERATIONMESSAGE.enum_types_by_name['Action']
_CONNECTIONMESSAGE_ACTION = _CONNECTIONMESSAGE.enum_types_by_name['Action']
Wrapper = _reflection.GeneratedProtocolMessageType('Wrapper', (_message.Message,), {
  'DESCRIPTOR' : _WRAPPER,
  '__module__' : 'nebula_pb2'
  # @@protoc_insertion_point(class_scope:nebula.Wrapper)
  })
_sym_db.RegisterMessage(Wrapper)

DiscoveryMessage = _reflection.GeneratedProtocolMessageType('DiscoveryMessage', (_message.Message,), {
  'DESCRIPTOR' : _DISCOVERYMESSAGE,
  '__module__' : 'nebula_pb2'
  # @@protoc_insertion_point(class_scope:nebula.DiscoveryMessage)
  })
_sym_db.RegisterMessage(DiscoveryMessage)

ControlMessage = _reflection.GeneratedProtocolMessageType('ControlMessage', (_message.Message,), {
  'DESCRIPTOR' : _CONTROLMESSAGE,
  '__module__' : 'nebula_pb2'
  # @@protoc_insertion_point(class_scope:nebula.ControlMessage)
  })
_sym_db.RegisterMessage(ControlMessage)

FederationMessage = _reflection.GeneratedProtocolMessageType('FederationMessage', (_message.Message,), {
  'DESCRIPTOR' : _FEDERATIONMESSAGE,
  '__module__' : 'nebula_pb2'
  # @@protoc_insertion_point(class_scope:nebula.FederationMessage)
  })
_sym_db.RegisterMessage(FederationMessage)

ModelMessage = _reflection.GeneratedProtocolMessageType('ModelMessage', (_message.Message,), {
  'DESCRIPTOR' : _MODELMESSAGE,
  '__module__' : 'nebula_pb2'
  # @@protoc_insertion_point(class_scope:nebula.ModelMessage)
  })
_sym_db.RegisterMessage(ModelMessage)

ConnectionMessage = _reflection.GeneratedProtocolMessageType('ConnectionMessage', (_message.Message,), {
  'DESCRIPTOR' : _CONNECTIONMESSAGE,
  '__module__' : 'nebula_pb2'
  # @@protoc_insertion_point(class_scope:nebula.ConnectionMessage)
  })
_sym_db.RegisterMessage(ConnectionMessage)

ResponseMessage = _reflection.GeneratedProtocolMessageType('ResponseMessage', (_message.Message,), {
  'DESCRIPTOR' : _RESPONSEMESSAGE,
  '__module__' : 'nebula_pb2'
  # @@protoc_insertion_point(class_scope:nebula.ResponseMessage)
  })
_sym_db.RegisterMessage(ResponseMessage)

NSSFeaturesMessage = _reflection.GeneratedProtocolMessageType('NSSFeaturesMessage', (_message.Message,), {
  'DESCRIPTOR' : _NSSFEATURESMESSAGE,
  '__module__' : 'nebula_pb2'
  # @@protoc_insertion_point(class_scope:nebula.NSSFeaturesMessage)
  })
_sym_db.RegisterMessage(NSSFeaturesMessage)

VoteMessage = _reflection.GeneratedProtocolMessageType('VoteMessage', (_message.Message,), {
  'DESCRIPTOR' : _VOTEMESSAGE,
  '__module__' : 'nebula_pb2'
  # @@protoc_insertion_point(class_scope:nebula.VoteMessage)
  })
_sym_db.RegisterMessage(VoteMessage)

if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _WRAPPER._serialized_start=25
  _WRAPPER._serialized_end=486
  _DISCOVERYMESSAGE._serialized_start=489
  _DISCOVERYMESSAGE._serialized_end=647
  _DISCOVERYMESSAGE_ACTION._serialized_start=595
  _DISCOVERYMESSAGE_ACTION._serialized_end=647
  _CONTROLMESSAGE._serialized_start=650
  _CONTROLMESSAGE._serialized_end=804
  _CONTROLMESSAGE_ACTION._serialized_start=728
  _CONTROLMESSAGE_ACTION._serialized_end=804
  _FEDERATIONMESSAGE._serialized_start=807
  _FEDERATIONMESSAGE._serialized_end=1012
  _FEDERATIONMESSAGE_ACTION._serialized_start=912
  _FEDERATIONMESSAGE_ACTION._serialized_end=1012
  _MODELMESSAGE._serialized_start=1014
  _MODELMESSAGE._serialized_end=1079
  _CONNECTIONMESSAGE._serialized_start=1081
  _CONNECTIONMESSAGE._serialized_end=1189
  _CONNECTIONMESSAGE_ACTION._serialized_start=1152
  _CONNECTIONMESSAGE_ACTION._serialized_end=1189
  _RESPONSEMESSAGE._serialized_start=1191
  _RESPONSEMESSAGE._serialized_end=1226
  _NSSFEATURESMESSAGE._serialized_start=1229
  _NSSFEATURESMESSAGE._serialized_end=1371
  _VOTEMESSAGE._serialized_start=1373
  _VOTEMESSAGE._serialized_end=1400
# @@protoc_insertion_point(module_scope)
