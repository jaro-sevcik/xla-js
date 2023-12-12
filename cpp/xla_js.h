#ifndef XLA_JS_H_
#define XLA_JS_H_

#include <napi.h>

#include <iostream>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wuninitialized"
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#pragma GCC diagnostic ignored "-Winvalid-offsetof"
#pragma GCC diagnostic ignored "-Wreturn-type"
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wdeprecated-copy"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wtype-limits"
#pragma GCC diagnostic ignored "-Wcomment"
#pragma GCC diagnostic ignored "-Wbuiltin-macro-redefined"

#define LLVM_ON_UNIX = 1
#define LLVM_VERSION_STRING

#include "xla/client/client_library.h"
#include "xla/client/xla_builder.h"
#include "xla/literal_util.h"
#include "xla/pjrt/tfrt_cpu_pjrt_client.h"
#include "xla/shape_util.h"

class PjRtClient : public Napi::ObjectWrap<PjRtClient> {
public:
  static Napi::Object Init(Napi::Env env, Napi::Object exports);
  PjRtClient(const Napi::CallbackInfo &info);
  Napi::Value Compile(const Napi::CallbackInfo &info);
  Napi::Value BufferFromHostLiteral(const Napi::CallbackInfo &info);

  xla::PjRtClient *value() { return client_.get(); }

private:
  std::unique_ptr<xla::PjRtClient> client_;
};

template <class T, typename TPtr>
class ReferenceWrapper : public Napi::ObjectWrap<T> {
public:
  ReferenceWrapper(const Napi::CallbackInfo &info, TPtr *ptr = nullptr);

  static Napi::Object
  Init(Napi::Env env, Napi::Object exports, const char *name,
       const std::initializer_list<Napi::ClassPropertyDescriptor<T>>
           &properties);
  static Napi::Object Create(Napi::Env env, TPtr *op);
  static TPtr *FromValue(Napi::Env env, Napi::Value value);

  TPtr *value() { return value_.get(); }

protected:
  static Napi::FunctionReference constructor;
  static napi_type_tag type_tag_;

  std::unique_ptr<TPtr> value_;
};

class XlaBuilderWrapper
    : public ReferenceWrapper<XlaBuilderWrapper, xla::XlaBuilder> {
public:
  static Napi::Object Init(Napi::Env env, Napi::Object exports);
  XlaBuilderWrapper(const Napi::CallbackInfo &info)
      : ReferenceWrapper<XlaBuilderWrapper, xla::XlaBuilder>(
            info, createBuilder(info)) {}

  Napi::Value Build(const Napi::CallbackInfo &info);

private:
  static xla::XlaBuilder *createBuilder(const Napi::CallbackInfo &info);
};

class XlaOpWrapper : public ReferenceWrapper<XlaOpWrapper, xla::XlaOp> {
public:
  static Napi::Object Init(Napi::Env env, Napi::Object exports);
  XlaOpWrapper(const Napi::CallbackInfo &info)
      : ReferenceWrapper<XlaOpWrapper, xla::XlaOp>(info) {}

  Napi::Value Dump(const Napi::CallbackInfo &info);
};

class XlaComputationWrapper
    : public ReferenceWrapper<XlaComputationWrapper, xla::XlaComputation> {
public:
  static Napi::Object Init(Napi::Env env, Napi::Object exports);
  XlaComputationWrapper(const Napi::CallbackInfo &info)
      : ReferenceWrapper<XlaComputationWrapper, xla::XlaComputation>(info) {}
};

class PjRtLoadedExecutableWrapper
    : public ReferenceWrapper<PjRtLoadedExecutableWrapper,
                              xla::PjRtLoadedExecutable> {
public:
  static Napi::Object Init(Napi::Env env, Napi::Object exports);
  Napi::Value Execute(const Napi::CallbackInfo &info);
  PjRtLoadedExecutableWrapper(const Napi::CallbackInfo &info)
      : ReferenceWrapper<PjRtLoadedExecutableWrapper,
                         xla::PjRtLoadedExecutable>(info) {}
};

class PjRtBufferWrapper
    : public ReferenceWrapper<PjRtBufferWrapper, xla::PjRtBuffer> {
public:
  static Napi::Object Init(Napi::Env env, Napi::Object exports);
  PjRtBufferWrapper(const Napi::CallbackInfo &info)
      : ReferenceWrapper<PjRtBufferWrapper, xla::PjRtBuffer>(info) {}

  Napi::Value ToLiteralSync(const Napi::CallbackInfo &info);
};

class LiteralWrapper : public ReferenceWrapper<LiteralWrapper, xla::Literal> {
public:
  static Napi::Object Init(Napi::Env env, Napi::Object exports);
  LiteralWrapper(const Napi::CallbackInfo &info)
      : ReferenceWrapper<LiteralWrapper, xla::Literal>(info) {}

  Napi::Value GetFirstElementF32(const Napi::CallbackInfo &info);
  Napi::Value Data(const Napi::CallbackInfo &info);
  Napi::Value Shape(const Napi::CallbackInfo &info);
  Napi::Value Reshape(const Napi::CallbackInfo &info);
  Napi::Value Broadcast(const Napi::CallbackInfo &info);
  Napi::Value ToString(const Napi::CallbackInfo &info);

  static Napi::Value CreateR0(const Napi::CallbackInfo &info);
  static Napi::Value CreateR1(const Napi::CallbackInfo &info);
};

#define PRIMITIVE_TYPE_V(V)                                                    \
  V(PRED)                                                                      \
  V(S4)                                                                        \
  V(S8)                                                                        \
  V(S16)                                                                       \
  V(S32)                                                                       \
  V(S64)                                                                       \
  V(U4)                                                                        \
  V(U8)                                                                        \
  V(U16)                                                                       \
  V(U32)                                                                       \
  V(U64)                                                                       \
  V(F16)                                                                       \
  V(F32)                                                                       \
  V(BF16)                                                                      \
  V(F64)                                                                       \
  V(F8E5M2)                                                                    \
  V(F8E4M3FN)                                                                  \
  V(F8E4M3B11FNUZ)                                                             \
  V(F8E5M2FNUZ)                                                                \
  V(F8E4M3FNUZ)

class ShapeWrapper : public ReferenceWrapper<ShapeWrapper, xla::Shape> {
public:
  static Napi::Object Init(Napi::Env env, Napi::Object exports);
  ShapeWrapper(const Napi::CallbackInfo &info)
      : ReferenceWrapper<ShapeWrapper, xla::Shape>(info) {}

  Napi::Value Dimensions(const Napi::CallbackInfo &info);
  Napi::Value PrimitiveType(const Napi::CallbackInfo &info);

  static Napi::Value ForArray(const Napi::CallbackInfo &info);
};

#endif // XLA_JS_H_
