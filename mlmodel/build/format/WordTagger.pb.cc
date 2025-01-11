// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: WordTagger.proto

#include "WordTagger.pb.h"

#include <algorithm>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/extension_set.h>
#include <google/protobuf/wire_format_lite.h>
#include <google/protobuf/io/zero_copy_stream_impl_lite.h>
// @@protoc_insertion_point(includes)
#include <google/protobuf/port_def.inc>

PROTOBUF_PRAGMA_INIT_SEG
namespace CoreML {
namespace Specification {
namespace CoreMLModels {
constexpr WordTagger::WordTagger(
  ::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized)
  : language_(&::PROTOBUF_NAMESPACE_ID::internal::fixed_address_empty_string)
  , tokensoutputfeaturename_(&::PROTOBUF_NAMESPACE_ID::internal::fixed_address_empty_string)
  , tokentagsoutputfeaturename_(&::PROTOBUF_NAMESPACE_ID::internal::fixed_address_empty_string)
  , tokenlocationsoutputfeaturename_(&::PROTOBUF_NAMESPACE_ID::internal::fixed_address_empty_string)
  , tokenlengthsoutputfeaturename_(&::PROTOBUF_NAMESPACE_ID::internal::fixed_address_empty_string)
  , modelparameterdata_(&::PROTOBUF_NAMESPACE_ID::internal::fixed_address_empty_string)
  , revision_(0u)
  , _oneof_case_{}{}
struct WordTaggerDefaultTypeInternal {
  constexpr WordTaggerDefaultTypeInternal()
    : _instance(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized{}) {}
  ~WordTaggerDefaultTypeInternal() {}
  union {
    WordTagger _instance;
  };
};
PROTOBUF_ATTRIBUTE_NO_DESTROY PROTOBUF_CONSTINIT WordTaggerDefaultTypeInternal _WordTagger_default_instance_;
}  // namespace CoreMLModels
}  // namespace Specification
}  // namespace CoreML
namespace CoreML {
namespace Specification {
namespace CoreMLModels {

// ===================================================================

class WordTagger::_Internal {
 public:
  static const ::CoreML::Specification::StringVector& stringtags(const WordTagger* msg);
};

const ::CoreML::Specification::StringVector&
WordTagger::_Internal::stringtags(const WordTagger* msg) {
  return *msg->Tags_.stringtags_;
}
void WordTagger::set_allocated_stringtags(::CoreML::Specification::StringVector* stringtags) {
  ::PROTOBUF_NAMESPACE_ID::Arena* message_arena = GetArenaForAllocation();
  clear_Tags();
  if (stringtags) {
    ::PROTOBUF_NAMESPACE_ID::Arena* submessage_arena =
        ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper<
            ::PROTOBUF_NAMESPACE_ID::MessageLite>::GetOwningArena(
                reinterpret_cast<::PROTOBUF_NAMESPACE_ID::MessageLite*>(stringtags));
    if (message_arena != submessage_arena) {
      stringtags = ::PROTOBUF_NAMESPACE_ID::internal::GetOwnedMessage(
          message_arena, stringtags, submessage_arena);
    }
    set_has_stringtags();
    Tags_.stringtags_ = stringtags;
  }
  // @@protoc_insertion_point(field_set_allocated:CoreML.Specification.CoreMLModels.WordTagger.stringTags)
}
void WordTagger::clear_stringtags() {
  if (_internal_has_stringtags()) {
    if (GetArenaForAllocation() == nullptr) {
      delete Tags_.stringtags_;
    }
    clear_has_Tags();
  }
}
WordTagger::WordTagger(::PROTOBUF_NAMESPACE_ID::Arena* arena,
                         bool is_message_owned)
  : ::PROTOBUF_NAMESPACE_ID::MessageLite(arena, is_message_owned) {
  SharedCtor();
  if (!is_message_owned) {
    RegisterArenaDtor(arena);
  }
  // @@protoc_insertion_point(arena_constructor:CoreML.Specification.CoreMLModels.WordTagger)
}
WordTagger::WordTagger(const WordTagger& from)
  : ::PROTOBUF_NAMESPACE_ID::MessageLite() {
  _internal_metadata_.MergeFrom<std::string>(from._internal_metadata_);
  language_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  #ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
    language_.Set(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), "", GetArenaForAllocation());
  #endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
  if (!from._internal_language().empty()) {
    language_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, from._internal_language(), 
      GetArenaForAllocation());
  }
  tokensoutputfeaturename_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  #ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
    tokensoutputfeaturename_.Set(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), "", GetArenaForAllocation());
  #endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
  if (!from._internal_tokensoutputfeaturename().empty()) {
    tokensoutputfeaturename_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, from._internal_tokensoutputfeaturename(), 
      GetArenaForAllocation());
  }
  tokentagsoutputfeaturename_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  #ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
    tokentagsoutputfeaturename_.Set(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), "", GetArenaForAllocation());
  #endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
  if (!from._internal_tokentagsoutputfeaturename().empty()) {
    tokentagsoutputfeaturename_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, from._internal_tokentagsoutputfeaturename(), 
      GetArenaForAllocation());
  }
  tokenlocationsoutputfeaturename_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  #ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
    tokenlocationsoutputfeaturename_.Set(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), "", GetArenaForAllocation());
  #endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
  if (!from._internal_tokenlocationsoutputfeaturename().empty()) {
    tokenlocationsoutputfeaturename_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, from._internal_tokenlocationsoutputfeaturename(), 
      GetArenaForAllocation());
  }
  tokenlengthsoutputfeaturename_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  #ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
    tokenlengthsoutputfeaturename_.Set(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), "", GetArenaForAllocation());
  #endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
  if (!from._internal_tokenlengthsoutputfeaturename().empty()) {
    tokenlengthsoutputfeaturename_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, from._internal_tokenlengthsoutputfeaturename(), 
      GetArenaForAllocation());
  }
  modelparameterdata_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  #ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
    modelparameterdata_.Set(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), "", GetArenaForAllocation());
  #endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
  if (!from._internal_modelparameterdata().empty()) {
    modelparameterdata_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, from._internal_modelparameterdata(), 
      GetArenaForAllocation());
  }
  revision_ = from.revision_;
  clear_has_Tags();
  switch (from.Tags_case()) {
    case kStringTags: {
      _internal_mutable_stringtags()->::CoreML::Specification::StringVector::MergeFrom(from._internal_stringtags());
      break;
    }
    case TAGS_NOT_SET: {
      break;
    }
  }
  // @@protoc_insertion_point(copy_constructor:CoreML.Specification.CoreMLModels.WordTagger)
}

inline void WordTagger::SharedCtor() {
language_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
#ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
  language_.Set(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), "", GetArenaForAllocation());
#endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
tokensoutputfeaturename_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
#ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
  tokensoutputfeaturename_.Set(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), "", GetArenaForAllocation());
#endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
tokentagsoutputfeaturename_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
#ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
  tokentagsoutputfeaturename_.Set(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), "", GetArenaForAllocation());
#endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
tokenlocationsoutputfeaturename_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
#ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
  tokenlocationsoutputfeaturename_.Set(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), "", GetArenaForAllocation());
#endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
tokenlengthsoutputfeaturename_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
#ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
  tokenlengthsoutputfeaturename_.Set(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), "", GetArenaForAllocation());
#endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
modelparameterdata_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
#ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
  modelparameterdata_.Set(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), "", GetArenaForAllocation());
#endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
revision_ = 0u;
clear_has_Tags();
}

WordTagger::~WordTagger() {
  // @@protoc_insertion_point(destructor:CoreML.Specification.CoreMLModels.WordTagger)
  if (GetArenaForAllocation() != nullptr) return;
  SharedDtor();
  _internal_metadata_.Delete<std::string>();
}

inline void WordTagger::SharedDtor() {
  GOOGLE_DCHECK(GetArenaForAllocation() == nullptr);
  language_.DestroyNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  tokensoutputfeaturename_.DestroyNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  tokentagsoutputfeaturename_.DestroyNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  tokenlocationsoutputfeaturename_.DestroyNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  tokenlengthsoutputfeaturename_.DestroyNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  modelparameterdata_.DestroyNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  if (has_Tags()) {
    clear_Tags();
  }
}

void WordTagger::ArenaDtor(void* object) {
  WordTagger* _this = reinterpret_cast< WordTagger* >(object);
  (void)_this;
}
void WordTagger::RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena*) {
}
void WordTagger::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}

void WordTagger::clear_Tags() {
// @@protoc_insertion_point(one_of_clear_start:CoreML.Specification.CoreMLModels.WordTagger)
  switch (Tags_case()) {
    case kStringTags: {
      if (GetArenaForAllocation() == nullptr) {
        delete Tags_.stringtags_;
      }
      break;
    }
    case TAGS_NOT_SET: {
      break;
    }
  }
  _oneof_case_[0] = TAGS_NOT_SET;
}


void WordTagger::Clear() {
// @@protoc_insertion_point(message_clear_start:CoreML.Specification.CoreMLModels.WordTagger)
  uint32_t cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  language_.ClearToEmpty();
  tokensoutputfeaturename_.ClearToEmpty();
  tokentagsoutputfeaturename_.ClearToEmpty();
  tokenlocationsoutputfeaturename_.ClearToEmpty();
  tokenlengthsoutputfeaturename_.ClearToEmpty();
  modelparameterdata_.ClearToEmpty();
  revision_ = 0u;
  clear_Tags();
  _internal_metadata_.Clear<std::string>();
}

const char* WordTagger::_InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  while (!ctx->Done(&ptr)) {
    uint32_t tag;
    ptr = ::PROTOBUF_NAMESPACE_ID::internal::ReadTag(ptr, &tag);
    switch (tag >> 3) {
      // uint32 revision = 1;
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 8)) {
          revision_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint32(&ptr);
          CHK_(ptr);
        } else
          goto handle_unusual;
        continue;
      // string language = 10;
      case 10:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 82)) {
          auto str = _internal_mutable_language();
          ptr = ::PROTOBUF_NAMESPACE_ID::internal::InlineGreedyStringParser(str, ptr, ctx);
          CHK_(::PROTOBUF_NAMESPACE_ID::internal::VerifyUTF8(str, nullptr));
          CHK_(ptr);
        } else
          goto handle_unusual;
        continue;
      // string tokensOutputFeatureName = 20;
      case 20:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 162)) {
          auto str = _internal_mutable_tokensoutputfeaturename();
          ptr = ::PROTOBUF_NAMESPACE_ID::internal::InlineGreedyStringParser(str, ptr, ctx);
          CHK_(::PROTOBUF_NAMESPACE_ID::internal::VerifyUTF8(str, nullptr));
          CHK_(ptr);
        } else
          goto handle_unusual;
        continue;
      // string tokenTagsOutputFeatureName = 21;
      case 21:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 170)) {
          auto str = _internal_mutable_tokentagsoutputfeaturename();
          ptr = ::PROTOBUF_NAMESPACE_ID::internal::InlineGreedyStringParser(str, ptr, ctx);
          CHK_(::PROTOBUF_NAMESPACE_ID::internal::VerifyUTF8(str, nullptr));
          CHK_(ptr);
        } else
          goto handle_unusual;
        continue;
      // string tokenLocationsOutputFeatureName = 22;
      case 22:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 178)) {
          auto str = _internal_mutable_tokenlocationsoutputfeaturename();
          ptr = ::PROTOBUF_NAMESPACE_ID::internal::InlineGreedyStringParser(str, ptr, ctx);
          CHK_(::PROTOBUF_NAMESPACE_ID::internal::VerifyUTF8(str, nullptr));
          CHK_(ptr);
        } else
          goto handle_unusual;
        continue;
      // string tokenLengthsOutputFeatureName = 23;
      case 23:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 186)) {
          auto str = _internal_mutable_tokenlengthsoutputfeaturename();
          ptr = ::PROTOBUF_NAMESPACE_ID::internal::InlineGreedyStringParser(str, ptr, ctx);
          CHK_(::PROTOBUF_NAMESPACE_ID::internal::VerifyUTF8(str, nullptr));
          CHK_(ptr);
        } else
          goto handle_unusual;
        continue;
      // bytes modelParameterData = 100;
      case 100:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 34)) {
          auto str = _internal_mutable_modelparameterdata();
          ptr = ::PROTOBUF_NAMESPACE_ID::internal::InlineGreedyStringParser(str, ptr, ctx);
          CHK_(ptr);
        } else
          goto handle_unusual;
        continue;
      // .CoreML.Specification.StringVector stringTags = 200;
      case 200:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 66)) {
          ptr = ctx->ParseMessage(_internal_mutable_stringtags(), ptr);
          CHK_(ptr);
        } else
          goto handle_unusual;
        continue;
      default:
        goto handle_unusual;
    }  // switch
  handle_unusual:
    if ((tag == 0) || ((tag & 7) == 4)) {
      CHK_(ptr);
      ctx->SetLastTag(tag);
      goto message_done;
    }
    ptr = UnknownFieldParse(
        tag,
        _internal_metadata_.mutable_unknown_fields<std::string>(),
        ptr, ctx);
    CHK_(ptr != nullptr);
  }  // while
message_done:
  return ptr;
failure:
  ptr = nullptr;
  goto message_done;
#undef CHK_
}

uint8_t* WordTagger::_InternalSerialize(
    uint8_t* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:CoreML.Specification.CoreMLModels.WordTagger)
  uint32_t cached_has_bits = 0;
  (void) cached_has_bits;

  // uint32 revision = 1;
  if (this->_internal_revision() != 0) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteUInt32ToArray(1, this->_internal_revision(), target);
  }

  // string language = 10;
  if (!this->_internal_language().empty()) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::VerifyUtf8String(
      this->_internal_language().data(), static_cast<int>(this->_internal_language().length()),
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::SERIALIZE,
      "CoreML.Specification.CoreMLModels.WordTagger.language");
    target = stream->WriteStringMaybeAliased(
        10, this->_internal_language(), target);
  }

  // string tokensOutputFeatureName = 20;
  if (!this->_internal_tokensoutputfeaturename().empty()) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::VerifyUtf8String(
      this->_internal_tokensoutputfeaturename().data(), static_cast<int>(this->_internal_tokensoutputfeaturename().length()),
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::SERIALIZE,
      "CoreML.Specification.CoreMLModels.WordTagger.tokensOutputFeatureName");
    target = stream->WriteStringMaybeAliased(
        20, this->_internal_tokensoutputfeaturename(), target);
  }

  // string tokenTagsOutputFeatureName = 21;
  if (!this->_internal_tokentagsoutputfeaturename().empty()) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::VerifyUtf8String(
      this->_internal_tokentagsoutputfeaturename().data(), static_cast<int>(this->_internal_tokentagsoutputfeaturename().length()),
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::SERIALIZE,
      "CoreML.Specification.CoreMLModels.WordTagger.tokenTagsOutputFeatureName");
    target = stream->WriteStringMaybeAliased(
        21, this->_internal_tokentagsoutputfeaturename(), target);
  }

  // string tokenLocationsOutputFeatureName = 22;
  if (!this->_internal_tokenlocationsoutputfeaturename().empty()) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::VerifyUtf8String(
      this->_internal_tokenlocationsoutputfeaturename().data(), static_cast<int>(this->_internal_tokenlocationsoutputfeaturename().length()),
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::SERIALIZE,
      "CoreML.Specification.CoreMLModels.WordTagger.tokenLocationsOutputFeatureName");
    target = stream->WriteStringMaybeAliased(
        22, this->_internal_tokenlocationsoutputfeaturename(), target);
  }

  // string tokenLengthsOutputFeatureName = 23;
  if (!this->_internal_tokenlengthsoutputfeaturename().empty()) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::VerifyUtf8String(
      this->_internal_tokenlengthsoutputfeaturename().data(), static_cast<int>(this->_internal_tokenlengthsoutputfeaturename().length()),
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::SERIALIZE,
      "CoreML.Specification.CoreMLModels.WordTagger.tokenLengthsOutputFeatureName");
    target = stream->WriteStringMaybeAliased(
        23, this->_internal_tokenlengthsoutputfeaturename(), target);
  }

  // bytes modelParameterData = 100;
  if (!this->_internal_modelparameterdata().empty()) {
    target = stream->WriteBytesMaybeAliased(
        100, this->_internal_modelparameterdata(), target);
  }

  // .CoreML.Specification.StringVector stringTags = 200;
  if (_internal_has_stringtags()) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::
      InternalWriteMessage(
        200, _Internal::stringtags(this), target, stream);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = stream->WriteRaw(_internal_metadata_.unknown_fields<std::string>(::PROTOBUF_NAMESPACE_ID::internal::GetEmptyString).data(),
        static_cast<int>(_internal_metadata_.unknown_fields<std::string>(::PROTOBUF_NAMESPACE_ID::internal::GetEmptyString).size()), target);
  }
  // @@protoc_insertion_point(serialize_to_array_end:CoreML.Specification.CoreMLModels.WordTagger)
  return target;
}

size_t WordTagger::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:CoreML.Specification.CoreMLModels.WordTagger)
  size_t total_size = 0;

  uint32_t cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  // string language = 10;
  if (!this->_internal_language().empty()) {
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::StringSize(
        this->_internal_language());
  }

  // string tokensOutputFeatureName = 20;
  if (!this->_internal_tokensoutputfeaturename().empty()) {
    total_size += 2 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::StringSize(
        this->_internal_tokensoutputfeaturename());
  }

  // string tokenTagsOutputFeatureName = 21;
  if (!this->_internal_tokentagsoutputfeaturename().empty()) {
    total_size += 2 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::StringSize(
        this->_internal_tokentagsoutputfeaturename());
  }

  // string tokenLocationsOutputFeatureName = 22;
  if (!this->_internal_tokenlocationsoutputfeaturename().empty()) {
    total_size += 2 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::StringSize(
        this->_internal_tokenlocationsoutputfeaturename());
  }

  // string tokenLengthsOutputFeatureName = 23;
  if (!this->_internal_tokenlengthsoutputfeaturename().empty()) {
    total_size += 2 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::StringSize(
        this->_internal_tokenlengthsoutputfeaturename());
  }

  // bytes modelParameterData = 100;
  if (!this->_internal_modelparameterdata().empty()) {
    total_size += 2 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::BytesSize(
        this->_internal_modelparameterdata());
  }

  // uint32 revision = 1;
  if (this->_internal_revision() != 0) {
    total_size += ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::UInt32SizePlusOne(this->_internal_revision());
  }

  switch (Tags_case()) {
    // .CoreML.Specification.StringVector stringTags = 200;
    case kStringTags: {
      total_size += 2 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::MessageSize(
          *Tags_.stringtags_);
      break;
    }
    case TAGS_NOT_SET: {
      break;
    }
  }
  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    total_size += _internal_metadata_.unknown_fields<std::string>(::PROTOBUF_NAMESPACE_ID::internal::GetEmptyString).size();
  }
  int cached_size = ::PROTOBUF_NAMESPACE_ID::internal::ToCachedSize(total_size);
  SetCachedSize(cached_size);
  return total_size;
}

void WordTagger::CheckTypeAndMergeFrom(
    const ::PROTOBUF_NAMESPACE_ID::MessageLite& from) {
  MergeFrom(*::PROTOBUF_NAMESPACE_ID::internal::DownCast<const WordTagger*>(
      &from));
}

void WordTagger::MergeFrom(const WordTagger& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:CoreML.Specification.CoreMLModels.WordTagger)
  GOOGLE_DCHECK_NE(&from, this);
  uint32_t cached_has_bits = 0;
  (void) cached_has_bits;

  if (!from._internal_language().empty()) {
    _internal_set_language(from._internal_language());
  }
  if (!from._internal_tokensoutputfeaturename().empty()) {
    _internal_set_tokensoutputfeaturename(from._internal_tokensoutputfeaturename());
  }
  if (!from._internal_tokentagsoutputfeaturename().empty()) {
    _internal_set_tokentagsoutputfeaturename(from._internal_tokentagsoutputfeaturename());
  }
  if (!from._internal_tokenlocationsoutputfeaturename().empty()) {
    _internal_set_tokenlocationsoutputfeaturename(from._internal_tokenlocationsoutputfeaturename());
  }
  if (!from._internal_tokenlengthsoutputfeaturename().empty()) {
    _internal_set_tokenlengthsoutputfeaturename(from._internal_tokenlengthsoutputfeaturename());
  }
  if (!from._internal_modelparameterdata().empty()) {
    _internal_set_modelparameterdata(from._internal_modelparameterdata());
  }
  if (from._internal_revision() != 0) {
    _internal_set_revision(from._internal_revision());
  }
  switch (from.Tags_case()) {
    case kStringTags: {
      _internal_mutable_stringtags()->::CoreML::Specification::StringVector::MergeFrom(from._internal_stringtags());
      break;
    }
    case TAGS_NOT_SET: {
      break;
    }
  }
  _internal_metadata_.MergeFrom<std::string>(from._internal_metadata_);
}

void WordTagger::CopyFrom(const WordTagger& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:CoreML.Specification.CoreMLModels.WordTagger)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool WordTagger::IsInitialized() const {
  return true;
}

void WordTagger::InternalSwap(WordTagger* other) {
  using std::swap;
  auto* lhs_arena = GetArenaForAllocation();
  auto* rhs_arena = other->GetArenaForAllocation();
  _internal_metadata_.InternalSwap(&other->_internal_metadata_);
  ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::InternalSwap(
      &::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(),
      &language_, lhs_arena,
      &other->language_, rhs_arena
  );
  ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::InternalSwap(
      &::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(),
      &tokensoutputfeaturename_, lhs_arena,
      &other->tokensoutputfeaturename_, rhs_arena
  );
  ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::InternalSwap(
      &::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(),
      &tokentagsoutputfeaturename_, lhs_arena,
      &other->tokentagsoutputfeaturename_, rhs_arena
  );
  ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::InternalSwap(
      &::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(),
      &tokenlocationsoutputfeaturename_, lhs_arena,
      &other->tokenlocationsoutputfeaturename_, rhs_arena
  );
  ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::InternalSwap(
      &::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(),
      &tokenlengthsoutputfeaturename_, lhs_arena,
      &other->tokenlengthsoutputfeaturename_, rhs_arena
  );
  ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::InternalSwap(
      &::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(),
      &modelparameterdata_, lhs_arena,
      &other->modelparameterdata_, rhs_arena
  );
  swap(revision_, other->revision_);
  swap(Tags_, other->Tags_);
  swap(_oneof_case_[0], other->_oneof_case_[0]);
}

std::string WordTagger::GetTypeName() const {
  return "CoreML.Specification.CoreMLModels.WordTagger";
}


// @@protoc_insertion_point(namespace_scope)
}  // namespace CoreMLModels
}  // namespace Specification
}  // namespace CoreML
PROTOBUF_NAMESPACE_OPEN
template<> PROTOBUF_NOINLINE ::CoreML::Specification::CoreMLModels::WordTagger* Arena::CreateMaybeMessage< ::CoreML::Specification::CoreMLModels::WordTagger >(Arena* arena) {
  return Arena::CreateMessageInternal< ::CoreML::Specification::CoreMLModels::WordTagger >(arena);
}
PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)
#include <google/protobuf/port_undef.inc>
