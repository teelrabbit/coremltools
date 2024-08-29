// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: LinkedModel.proto

#ifndef PROTOBUF_LinkedModel_2eproto__INCLUDED
#define PROTOBUF_LinkedModel_2eproto__INCLUDED

#include <string>

#include <google/protobuf/stubs/common.h>

#if GOOGLE_PROTOBUF_VERSION < 3003000
#error This file was generated by a newer version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please update
#error your headers.
#endif
#if 3003000 < GOOGLE_PROTOBUF_MIN_PROTOC_VERSION
#error This file was generated by an older version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please
#error regenerate this file with a newer version of protoc.
#endif

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/arena.h>
#include <google/protobuf/arenastring.h>
#include <google/protobuf/generated_message_table_driven.h>
#include <google/protobuf/generated_message_util.h>
#include <google/protobuf/metadata_lite.h>
#include <google/protobuf/message_lite.h>
#include <google/protobuf/repeated_field.h>  // IWYU pragma: export
#include <google/protobuf/extension_set.h>  // IWYU pragma: export
#include "Parameters.pb.h"  // IWYU pragma: export
// @@protoc_insertion_point(includes)
namespace CoreML {
namespace Specification {
class ArrayFeatureType;
class ArrayFeatureTypeDefaultTypeInternal;
extern ArrayFeatureTypeDefaultTypeInternal _ArrayFeatureType_default_instance_;
class ArrayFeatureType_EnumeratedShapes;
class ArrayFeatureType_EnumeratedShapesDefaultTypeInternal;
extern ArrayFeatureType_EnumeratedShapesDefaultTypeInternal _ArrayFeatureType_EnumeratedShapes_default_instance_;
class ArrayFeatureType_Shape;
class ArrayFeatureType_ShapeDefaultTypeInternal;
extern ArrayFeatureType_ShapeDefaultTypeInternal _ArrayFeatureType_Shape_default_instance_;
class ArrayFeatureType_ShapeRange;
class ArrayFeatureType_ShapeRangeDefaultTypeInternal;
extern ArrayFeatureType_ShapeRangeDefaultTypeInternal _ArrayFeatureType_ShapeRange_default_instance_;
class BoolParameter;
class BoolParameterDefaultTypeInternal;
extern BoolParameterDefaultTypeInternal _BoolParameter_default_instance_;
class DictionaryFeatureType;
class DictionaryFeatureTypeDefaultTypeInternal;
extern DictionaryFeatureTypeDefaultTypeInternal _DictionaryFeatureType_default_instance_;
class DoubleFeatureType;
class DoubleFeatureTypeDefaultTypeInternal;
extern DoubleFeatureTypeDefaultTypeInternal _DoubleFeatureType_default_instance_;
class DoubleParameter;
class DoubleParameterDefaultTypeInternal;
extern DoubleParameterDefaultTypeInternal _DoubleParameter_default_instance_;
class DoubleRange;
class DoubleRangeDefaultTypeInternal;
extern DoubleRangeDefaultTypeInternal _DoubleRange_default_instance_;
class DoubleVector;
class DoubleVectorDefaultTypeInternal;
extern DoubleVectorDefaultTypeInternal _DoubleVector_default_instance_;
class FeatureType;
class FeatureTypeDefaultTypeInternal;
extern FeatureTypeDefaultTypeInternal _FeatureType_default_instance_;
class FloatVector;
class FloatVectorDefaultTypeInternal;
extern FloatVectorDefaultTypeInternal _FloatVector_default_instance_;
class ImageFeatureType;
class ImageFeatureTypeDefaultTypeInternal;
extern ImageFeatureTypeDefaultTypeInternal _ImageFeatureType_default_instance_;
class ImageFeatureType_EnumeratedImageSizes;
class ImageFeatureType_EnumeratedImageSizesDefaultTypeInternal;
extern ImageFeatureType_EnumeratedImageSizesDefaultTypeInternal _ImageFeatureType_EnumeratedImageSizes_default_instance_;
class ImageFeatureType_ImageSize;
class ImageFeatureType_ImageSizeDefaultTypeInternal;
extern ImageFeatureType_ImageSizeDefaultTypeInternal _ImageFeatureType_ImageSize_default_instance_;
class ImageFeatureType_ImageSizeRange;
class ImageFeatureType_ImageSizeRangeDefaultTypeInternal;
extern ImageFeatureType_ImageSizeRangeDefaultTypeInternal _ImageFeatureType_ImageSizeRange_default_instance_;
class Int64FeatureType;
class Int64FeatureTypeDefaultTypeInternal;
extern Int64FeatureTypeDefaultTypeInternal _Int64FeatureType_default_instance_;
class Int64Parameter;
class Int64ParameterDefaultTypeInternal;
extern Int64ParameterDefaultTypeInternal _Int64Parameter_default_instance_;
class Int64Range;
class Int64RangeDefaultTypeInternal;
extern Int64RangeDefaultTypeInternal _Int64Range_default_instance_;
class Int64Set;
class Int64SetDefaultTypeInternal;
extern Int64SetDefaultTypeInternal _Int64Set_default_instance_;
class Int64ToDoubleMap;
class Int64ToDoubleMapDefaultTypeInternal;
extern Int64ToDoubleMapDefaultTypeInternal _Int64ToDoubleMap_default_instance_;
class Int64ToDoubleMap_MapEntry;
class Int64ToDoubleMap_MapEntryDefaultTypeInternal;
extern Int64ToDoubleMap_MapEntryDefaultTypeInternal _Int64ToDoubleMap_MapEntry_default_instance_;
class Int64ToStringMap;
class Int64ToStringMapDefaultTypeInternal;
extern Int64ToStringMapDefaultTypeInternal _Int64ToStringMap_default_instance_;
class Int64ToStringMap_MapEntry;
class Int64ToStringMap_MapEntryDefaultTypeInternal;
extern Int64ToStringMap_MapEntryDefaultTypeInternal _Int64ToStringMap_MapEntry_default_instance_;
class Int64Vector;
class Int64VectorDefaultTypeInternal;
extern Int64VectorDefaultTypeInternal _Int64Vector_default_instance_;
class LinkedModel;
class LinkedModelDefaultTypeInternal;
extern LinkedModelDefaultTypeInternal _LinkedModel_default_instance_;
class LinkedModelFile;
class LinkedModelFileDefaultTypeInternal;
extern LinkedModelFileDefaultTypeInternal _LinkedModelFile_default_instance_;
class PrecisionRecallCurve;
class PrecisionRecallCurveDefaultTypeInternal;
extern PrecisionRecallCurveDefaultTypeInternal _PrecisionRecallCurve_default_instance_;
class SequenceFeatureType;
class SequenceFeatureTypeDefaultTypeInternal;
extern SequenceFeatureTypeDefaultTypeInternal _SequenceFeatureType_default_instance_;
class SizeRange;
class SizeRangeDefaultTypeInternal;
extern SizeRangeDefaultTypeInternal _SizeRange_default_instance_;
class StateFeatureType;
class StateFeatureTypeDefaultTypeInternal;
extern StateFeatureTypeDefaultTypeInternal _StateFeatureType_default_instance_;
class StringFeatureType;
class StringFeatureTypeDefaultTypeInternal;
extern StringFeatureTypeDefaultTypeInternal _StringFeatureType_default_instance_;
class StringParameter;
class StringParameterDefaultTypeInternal;
extern StringParameterDefaultTypeInternal _StringParameter_default_instance_;
class StringToDoubleMap;
class StringToDoubleMapDefaultTypeInternal;
extern StringToDoubleMapDefaultTypeInternal _StringToDoubleMap_default_instance_;
class StringToDoubleMap_MapEntry;
class StringToDoubleMap_MapEntryDefaultTypeInternal;
extern StringToDoubleMap_MapEntryDefaultTypeInternal _StringToDoubleMap_MapEntry_default_instance_;
class StringToInt64Map;
class StringToInt64MapDefaultTypeInternal;
extern StringToInt64MapDefaultTypeInternal _StringToInt64Map_default_instance_;
class StringToInt64Map_MapEntry;
class StringToInt64Map_MapEntryDefaultTypeInternal;
extern StringToInt64Map_MapEntryDefaultTypeInternal _StringToInt64Map_MapEntry_default_instance_;
class StringVector;
class StringVectorDefaultTypeInternal;
extern StringVectorDefaultTypeInternal _StringVector_default_instance_;
}  // namespace Specification
}  // namespace CoreML

namespace CoreML {
namespace Specification {

namespace protobuf_LinkedModel_2eproto {
// Internal implementation detail -- do not call these.
struct TableStruct {
  static const ::google::protobuf::internal::ParseTableField entries[];
  static const ::google::protobuf::internal::AuxillaryParseTableField aux[];
  static const ::google::protobuf::internal::ParseTable schema[];
  static const ::google::protobuf::uint32 offsets[];
  static void InitDefaultsImpl();
  static void Shutdown();
};
void AddDescriptors();
void InitDefaults();
}  // namespace protobuf_LinkedModel_2eproto

// ===================================================================

class LinkedModel : public ::google::protobuf::MessageLite /* @@protoc_insertion_point(class_definition:CoreML.Specification.LinkedModel) */ {
 public:
  LinkedModel();
  virtual ~LinkedModel();

  LinkedModel(const LinkedModel& from);

  inline LinkedModel& operator=(const LinkedModel& from) {
    CopyFrom(from);
    return *this;
  }

  static const LinkedModel& default_instance();

  enum LinkTypeCase {
    kLinkedModelFile = 1,
    LINKTYPE_NOT_SET = 0,
  };

  static inline const LinkedModel* internal_default_instance() {
    return reinterpret_cast<const LinkedModel*>(
               &_LinkedModel_default_instance_);
  }
  static PROTOBUF_CONSTEXPR int const kIndexInFileMessages =
    0;

  void Swap(LinkedModel* other);

  // implements Message ----------------------------------------------

  inline LinkedModel* New() const PROTOBUF_FINAL { return New(NULL); }

  LinkedModel* New(::google::protobuf::Arena* arena) const PROTOBUF_FINAL;
  void CheckTypeAndMergeFrom(const ::google::protobuf::MessageLite& from)
    PROTOBUF_FINAL;
  void CopyFrom(const LinkedModel& from);
  void MergeFrom(const LinkedModel& from);
  void Clear() PROTOBUF_FINAL;
  bool IsInitialized() const PROTOBUF_FINAL;

  size_t ByteSizeLong() const PROTOBUF_FINAL;
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input) PROTOBUF_FINAL;
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const PROTOBUF_FINAL;
  void DiscardUnknownFields();
  int GetCachedSize() const PROTOBUF_FINAL { return _cached_size_; }
  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const;
  void InternalSwap(LinkedModel* other);
  private:
  inline ::google::protobuf::Arena* GetArenaNoVirtual() const {
    return NULL;
  }
  inline void* MaybeArenaPtr() const {
    return NULL;
  }
  public:

  ::std::string GetTypeName() const PROTOBUF_FINAL;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // .CoreML.Specification.LinkedModelFile linkedModelFile = 1;
  bool has_linkedmodelfile() const;
  void clear_linkedmodelfile();
  static const int kLinkedModelFileFieldNumber = 1;
  const ::CoreML::Specification::LinkedModelFile& linkedmodelfile() const;
  ::CoreML::Specification::LinkedModelFile* mutable_linkedmodelfile();
  ::CoreML::Specification::LinkedModelFile* release_linkedmodelfile();
  void set_allocated_linkedmodelfile(::CoreML::Specification::LinkedModelFile* linkedmodelfile);

  LinkTypeCase LinkType_case() const;
  // @@protoc_insertion_point(class_scope:CoreML.Specification.LinkedModel)
 private:
  void set_has_linkedmodelfile();

  inline bool has_LinkType() const;
  void clear_LinkType();
  inline void clear_has_LinkType();

  ::google::protobuf::internal::InternalMetadataWithArenaLite _internal_metadata_;
  union LinkTypeUnion {
    LinkTypeUnion() {}
    ::CoreML::Specification::LinkedModelFile* linkedmodelfile_;
  } LinkType_;
  mutable int _cached_size_;
  ::google::protobuf::uint32 _oneof_case_[1];

  friend struct protobuf_LinkedModel_2eproto::TableStruct;
};
// -------------------------------------------------------------------

class LinkedModelFile : public ::google::protobuf::MessageLite /* @@protoc_insertion_point(class_definition:CoreML.Specification.LinkedModelFile) */ {
 public:
  LinkedModelFile();
  virtual ~LinkedModelFile();

  LinkedModelFile(const LinkedModelFile& from);

  inline LinkedModelFile& operator=(const LinkedModelFile& from) {
    CopyFrom(from);
    return *this;
  }

  static const LinkedModelFile& default_instance();

  static inline const LinkedModelFile* internal_default_instance() {
    return reinterpret_cast<const LinkedModelFile*>(
               &_LinkedModelFile_default_instance_);
  }
  static PROTOBUF_CONSTEXPR int const kIndexInFileMessages =
    1;

  void Swap(LinkedModelFile* other);

  // implements Message ----------------------------------------------

  inline LinkedModelFile* New() const PROTOBUF_FINAL { return New(NULL); }

  LinkedModelFile* New(::google::protobuf::Arena* arena) const PROTOBUF_FINAL;
  void CheckTypeAndMergeFrom(const ::google::protobuf::MessageLite& from)
    PROTOBUF_FINAL;
  void CopyFrom(const LinkedModelFile& from);
  void MergeFrom(const LinkedModelFile& from);
  void Clear() PROTOBUF_FINAL;
  bool IsInitialized() const PROTOBUF_FINAL;

  size_t ByteSizeLong() const PROTOBUF_FINAL;
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input) PROTOBUF_FINAL;
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const PROTOBUF_FINAL;
  void DiscardUnknownFields();
  int GetCachedSize() const PROTOBUF_FINAL { return _cached_size_; }
  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const;
  void InternalSwap(LinkedModelFile* other);
  private:
  inline ::google::protobuf::Arena* GetArenaNoVirtual() const {
    return NULL;
  }
  inline void* MaybeArenaPtr() const {
    return NULL;
  }
  public:

  ::std::string GetTypeName() const PROTOBUF_FINAL;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // .CoreML.Specification.StringParameter linkedModelFileName = 1;
  bool has_linkedmodelfilename() const;
  void clear_linkedmodelfilename();
  static const int kLinkedModelFileNameFieldNumber = 1;
  const ::CoreML::Specification::StringParameter& linkedmodelfilename() const;
  ::CoreML::Specification::StringParameter* mutable_linkedmodelfilename();
  ::CoreML::Specification::StringParameter* release_linkedmodelfilename();
  void set_allocated_linkedmodelfilename(::CoreML::Specification::StringParameter* linkedmodelfilename);

  // .CoreML.Specification.StringParameter linkedModelSearchPath = 2;
  bool has_linkedmodelsearchpath() const;
  void clear_linkedmodelsearchpath();
  static const int kLinkedModelSearchPathFieldNumber = 2;
  const ::CoreML::Specification::StringParameter& linkedmodelsearchpath() const;
  ::CoreML::Specification::StringParameter* mutable_linkedmodelsearchpath();
  ::CoreML::Specification::StringParameter* release_linkedmodelsearchpath();
  void set_allocated_linkedmodelsearchpath(::CoreML::Specification::StringParameter* linkedmodelsearchpath);

  // @@protoc_insertion_point(class_scope:CoreML.Specification.LinkedModelFile)
 private:

  ::google::protobuf::internal::InternalMetadataWithArenaLite _internal_metadata_;
  ::CoreML::Specification::StringParameter* linkedmodelfilename_;
  ::CoreML::Specification::StringParameter* linkedmodelsearchpath_;
  mutable int _cached_size_;
  friend struct protobuf_LinkedModel_2eproto::TableStruct;
};
// ===================================================================


// ===================================================================

#if !PROTOBUF_INLINE_NOT_IN_HEADERS
// LinkedModel

// .CoreML.Specification.LinkedModelFile linkedModelFile = 1;
inline bool LinkedModel::has_linkedmodelfile() const {
  return LinkType_case() == kLinkedModelFile;
}
inline void LinkedModel::set_has_linkedmodelfile() {
  _oneof_case_[0] = kLinkedModelFile;
}
inline void LinkedModel::clear_linkedmodelfile() {
  if (has_linkedmodelfile()) {
    delete LinkType_.linkedmodelfile_;
    clear_has_LinkType();
  }
}
inline  const ::CoreML::Specification::LinkedModelFile& LinkedModel::linkedmodelfile() const {
  // @@protoc_insertion_point(field_get:CoreML.Specification.LinkedModel.linkedModelFile)
  return has_linkedmodelfile()
      ? *LinkType_.linkedmodelfile_
      : ::CoreML::Specification::LinkedModelFile::default_instance();
}
inline ::CoreML::Specification::LinkedModelFile* LinkedModel::mutable_linkedmodelfile() {
  if (!has_linkedmodelfile()) {
    clear_LinkType();
    set_has_linkedmodelfile();
    LinkType_.linkedmodelfile_ = new ::CoreML::Specification::LinkedModelFile;
  }
  // @@protoc_insertion_point(field_mutable:CoreML.Specification.LinkedModel.linkedModelFile)
  return LinkType_.linkedmodelfile_;
}
inline ::CoreML::Specification::LinkedModelFile* LinkedModel::release_linkedmodelfile() {
  // @@protoc_insertion_point(field_release:CoreML.Specification.LinkedModel.linkedModelFile)
  if (has_linkedmodelfile()) {
    clear_has_LinkType();
    ::CoreML::Specification::LinkedModelFile* temp = LinkType_.linkedmodelfile_;
    LinkType_.linkedmodelfile_ = NULL;
    return temp;
  } else {
    return NULL;
  }
}
inline void LinkedModel::set_allocated_linkedmodelfile(::CoreML::Specification::LinkedModelFile* linkedmodelfile) {
  clear_LinkType();
  if (linkedmodelfile) {
    set_has_linkedmodelfile();
    LinkType_.linkedmodelfile_ = linkedmodelfile;
  }
  // @@protoc_insertion_point(field_set_allocated:CoreML.Specification.LinkedModel.linkedModelFile)
}

inline bool LinkedModel::has_LinkType() const {
  return LinkType_case() != LINKTYPE_NOT_SET;
}
inline void LinkedModel::clear_has_LinkType() {
  _oneof_case_[0] = LINKTYPE_NOT_SET;
}
inline LinkedModel::LinkTypeCase LinkedModel::LinkType_case() const {
  return LinkedModel::LinkTypeCase(_oneof_case_[0]);
}
// -------------------------------------------------------------------

// LinkedModelFile

// .CoreML.Specification.StringParameter linkedModelFileName = 1;
inline bool LinkedModelFile::has_linkedmodelfilename() const {
  return this != internal_default_instance() && linkedmodelfilename_ != NULL;
}
inline void LinkedModelFile::clear_linkedmodelfilename() {
  if (GetArenaNoVirtual() == NULL && linkedmodelfilename_ != NULL) delete linkedmodelfilename_;
  linkedmodelfilename_ = NULL;
}
inline const ::CoreML::Specification::StringParameter& LinkedModelFile::linkedmodelfilename() const {
  // @@protoc_insertion_point(field_get:CoreML.Specification.LinkedModelFile.linkedModelFileName)
  return linkedmodelfilename_ != NULL ? *linkedmodelfilename_
                         : *::CoreML::Specification::StringParameter::internal_default_instance();
}
inline ::CoreML::Specification::StringParameter* LinkedModelFile::mutable_linkedmodelfilename() {

  if (linkedmodelfilename_ == NULL) {
    linkedmodelfilename_ = new ::CoreML::Specification::StringParameter;
  }
  // @@protoc_insertion_point(field_mutable:CoreML.Specification.LinkedModelFile.linkedModelFileName)
  return linkedmodelfilename_;
}
inline ::CoreML::Specification::StringParameter* LinkedModelFile::release_linkedmodelfilename() {
  // @@protoc_insertion_point(field_release:CoreML.Specification.LinkedModelFile.linkedModelFileName)

  ::CoreML::Specification::StringParameter* temp = linkedmodelfilename_;
  linkedmodelfilename_ = NULL;
  return temp;
}
inline void LinkedModelFile::set_allocated_linkedmodelfilename(::CoreML::Specification::StringParameter* linkedmodelfilename) {
  delete linkedmodelfilename_;
  linkedmodelfilename_ = linkedmodelfilename;
  if (linkedmodelfilename) {

  } else {

  }
  // @@protoc_insertion_point(field_set_allocated:CoreML.Specification.LinkedModelFile.linkedModelFileName)
}

// .CoreML.Specification.StringParameter linkedModelSearchPath = 2;
inline bool LinkedModelFile::has_linkedmodelsearchpath() const {
  return this != internal_default_instance() && linkedmodelsearchpath_ != NULL;
}
inline void LinkedModelFile::clear_linkedmodelsearchpath() {
  if (GetArenaNoVirtual() == NULL && linkedmodelsearchpath_ != NULL) delete linkedmodelsearchpath_;
  linkedmodelsearchpath_ = NULL;
}
inline const ::CoreML::Specification::StringParameter& LinkedModelFile::linkedmodelsearchpath() const {
  // @@protoc_insertion_point(field_get:CoreML.Specification.LinkedModelFile.linkedModelSearchPath)
  return linkedmodelsearchpath_ != NULL ? *linkedmodelsearchpath_
                         : *::CoreML::Specification::StringParameter::internal_default_instance();
}
inline ::CoreML::Specification::StringParameter* LinkedModelFile::mutable_linkedmodelsearchpath() {

  if (linkedmodelsearchpath_ == NULL) {
    linkedmodelsearchpath_ = new ::CoreML::Specification::StringParameter;
  }
  // @@protoc_insertion_point(field_mutable:CoreML.Specification.LinkedModelFile.linkedModelSearchPath)
  return linkedmodelsearchpath_;
}
inline ::CoreML::Specification::StringParameter* LinkedModelFile::release_linkedmodelsearchpath() {
  // @@protoc_insertion_point(field_release:CoreML.Specification.LinkedModelFile.linkedModelSearchPath)

  ::CoreML::Specification::StringParameter* temp = linkedmodelsearchpath_;
  linkedmodelsearchpath_ = NULL;
  return temp;
}
inline void LinkedModelFile::set_allocated_linkedmodelsearchpath(::CoreML::Specification::StringParameter* linkedmodelsearchpath) {
  delete linkedmodelsearchpath_;
  linkedmodelsearchpath_ = linkedmodelsearchpath;
  if (linkedmodelsearchpath) {

  } else {

  }
  // @@protoc_insertion_point(field_set_allocated:CoreML.Specification.LinkedModelFile.linkedModelSearchPath)
}

#endif  // !PROTOBUF_INLINE_NOT_IN_HEADERS
// -------------------------------------------------------------------


// @@protoc_insertion_point(namespace_scope)


}  // namespace Specification
}  // namespace CoreML

// @@protoc_insertion_point(global_scope)

#endif  // PROTOBUF_LinkedModel_2eproto__INCLUDED
