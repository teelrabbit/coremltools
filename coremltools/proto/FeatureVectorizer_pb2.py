# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: FeatureVectorizer.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x17\x46\x65\x61tureVectorizer.proto\x12\x14\x43oreML.Specification\"\x98\x01\n\x11\x46\x65\x61tureVectorizer\x12\x46\n\tinputList\x18\x01 \x03(\x0b\x32\x33.CoreML.Specification.FeatureVectorizer.InputColumn\x1a;\n\x0bInputColumn\x12\x13\n\x0binputColumn\x18\x01 \x01(\t\x12\x17\n\x0finputDimensions\x18\x02 \x01(\x04\x42\x02H\x03\x62\x06proto3')



_FEATUREVECTORIZER = DESCRIPTOR.message_types_by_name['FeatureVectorizer']
_FEATUREVECTORIZER_INPUTCOLUMN = _FEATUREVECTORIZER.nested_types_by_name['InputColumn']
FeatureVectorizer = _reflection.GeneratedProtocolMessageType('FeatureVectorizer', (_message.Message,), {

  'InputColumn' : _reflection.GeneratedProtocolMessageType('InputColumn', (_message.Message,), {
    'DESCRIPTOR' : _FEATUREVECTORIZER_INPUTCOLUMN,
    '__module__' : 'FeatureVectorizer_pb2'
    # @@protoc_insertion_point(class_scope:CoreML.Specification.FeatureVectorizer.InputColumn)
    })
  ,
  'DESCRIPTOR' : _FEATUREVECTORIZER,
  '__module__' : 'FeatureVectorizer_pb2'
  # @@protoc_insertion_point(class_scope:CoreML.Specification.FeatureVectorizer)
  })
_sym_db.RegisterMessage(FeatureVectorizer)
_sym_db.RegisterMessage(FeatureVectorizer.InputColumn)

if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'H\003'
  _FEATUREVECTORIZER._serialized_start=50
  _FEATUREVECTORIZER._serialized_end=202
  _FEATUREVECTORIZER_INPUTCOLUMN._serialized_start=143
  _FEATUREVECTORIZER_INPUTCOLUMN._serialized_end=202
# @@protoc_insertion_point(module_scope)
