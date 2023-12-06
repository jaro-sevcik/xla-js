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

#define CONCAT(a, b) a##b

#define ASSIGN_OR_RETURN(name, rhs)                                            \
  ASSIGN_OR_RETURN_WITH_TEMP(CONCAT(name, __COUNTER__), name, rhs)

#define ASSIGN_OR_RETURN_WITH_TEMP(temp, name, rhs)                            \
  auto temp = rhs;                                                             \
  if (!temp.ok())                                                              \
    return temp.status();                                                      \
  auto name = std::move(temp.value());

// We need:
// - client
// - builder
// - literal
// - shape
// - buffer
// - op
// - executable
// - loaded executable
// - ?device

// client: createClient.
// builder: normal constructor.

class XlaClient : public Napi::ObjectWrap<XlaClient> {
public:
  static Napi::Object Init(Napi::Env env, Napi::Object exports);
  XlaClient(const Napi::CallbackInfo &info);
  Napi::Value Compile(const Napi::CallbackInfo &info);

private:
  std::unique_ptr<xla::PjRtClient> client_;
};

Napi::Value XlaClient::Compile(const Napi::CallbackInfo& info){
    Napi::Env env = info.Env();
    printf("Compiling!");
    return env.Null();
}

XlaClient::XlaClient(const Napi::CallbackInfo &info)
    : Napi::ObjectWrap<XlaClient>(info) {
  Napi::Env env = info.Env();

  auto client_status = xla::GetTfrtCpuClient(false);
  if (!client_status.ok()) {
    Napi::Error::New(env, "XLA client initialization failed").ThrowAsJavaScriptException();
    return;
  }

  this->client_ = std::move(client_status.value());
}

Napi::Object XlaClient::Init(Napi::Env env, Napi::Object exports) {
  Napi::Function func = DefineClass(
      env, "Client",
      {
          InstanceMethod(
              "compile", &XlaClient::Compile, static_cast<napi_property_attributes>(
                              napi_writable | napi_configurable)),
      });

  exports.Set("Client", func);
  return exports;
}

class XlaBuilderWrapper : public Napi::ObjectWrap<XlaBuilderWrapper> {
  public:
    static Napi::Object Init(Napi::Env env, Napi::Object exports);
    XlaBuilderWrapper(const Napi::CallbackInfo& info);
    Napi::Value Build(const Napi::CallbackInfo& info);


    static xla::XlaBuilder* BuilderFromValue(Napi::Env env, Napi::Value value);
  private:
    static napi_type_tag type_tag_;
    std::unique_ptr<xla::XlaBuilder> builder_;

    xla::XlaBuilder* builder() { return builder_.get(); }
};

napi_type_tag XlaBuilderWrapper::type_tag_ = { 0xdf6e6fbb6c724f60ULL, 0x842a52de6b318691ULL };

Napi::Object XlaBuilderWrapper::Init(Napi::Env env, Napi::Object exports) {
  Napi::Function func = DefineClass(
      env, "XlaBuilder",
      {
          InstanceMethod(
              "build", &XlaBuilderWrapper::Build, static_cast<napi_property_attributes>(
                              napi_writable | napi_configurable)),
      });

  exports.Set("XlaBuilder", func);
  return exports;
}

// static 
xla::XlaBuilder* XlaBuilderWrapper::BuilderFromValue(Napi::Env env, Napi::Value value) {
  if (!value.IsObject() || !value.As<Napi::Object>().CheckTypeTag(&type_tag_)) {
        Napi::Error::New(env, "XlaBuilder instance expected").ThrowAsJavaScriptException();
        return nullptr;
  }

  return XlaBuilderWrapper::Unwrap(value.As<Napi::Object>())->builder();
}

XlaBuilderWrapper::XlaBuilderWrapper(const Napi::CallbackInfo& info) : Napi::ObjectWrap<XlaBuilderWrapper>(info) {
  Napi::Env env = info.Env();

  info.This().As<Napi::Object>().TypeTag(&type_tag_);

  if (info.Length() != 1 || !info[0].IsString()) {
        Napi::Error::New(env, "XlaBuilder constructor expects string name as its single argument").ThrowAsJavaScriptException();
        return;
  }

  auto str = info[0].As<Napi::String>().Utf8Value();

  this->builder_.reset(new xla::XlaBuilder(str.c_str()));
}


Napi::Value XlaBuilderWrapper::Build(const Napi::CallbackInfo& info){
    Napi::Env env = info.Env();
    printf("Building!");
    return env.Null();
}

class XlaOpWrapper: public Napi::ObjectWrap<XlaOpWrapper> {
  public:
    static Napi::Object Init(Napi::Env env, Napi::Object exports);
    XlaOpWrapper(const Napi::CallbackInfo& info);

    Napi::Value Dump(const Napi::CallbackInfo& info);

    static Napi::Object create(Napi::Env env, xla::XlaOp *op);
  private:
    static Napi::FunctionReference constructor;
    static napi_type_tag type_tag_;

    std::unique_ptr<xla::XlaOp> op_;
};

napi_type_tag XlaOpWrapper::type_tag_ = { 0xdf6e6fbb6c724f60ULL, 0x842a52de6b318690ULL };

Napi::FunctionReference XlaOpWrapper::constructor;

// static 
Napi::Object XlaOpWrapper::create(Napi::Env env, xla::XlaOp *op) {
  auto r = Napi::External<xla::XlaOp>::New(env, op);
  r.TypeTag(&type_tag_);
  return XlaOpWrapper::constructor.New({r});

}

Napi::Object XlaOpWrapper::Init(Napi::Env env, Napi::Object exports) {
  Napi::Function func = DefineClass(
      env, "XlaOp",
      {
          InstanceMethod(
              "dump", &XlaOpWrapper::Dump, static_cast<napi_property_attributes>(
                              napi_writable | napi_configurable)),
      });

  constructor = Napi::Persistent(func);
  constructor.SuppressDestruct();
  exports.Set("XlaOp", func);
  return exports;
}

XlaOpWrapper::XlaOpWrapper(const Napi::CallbackInfo& info) : Napi::ObjectWrap<XlaOpWrapper>(info) {
  Napi::Env env = info.Env();
  if (info.Length() != 1 || !info[0].IsExternal()) {
        Napi::Error::New(env, "XlaOp can only be created with external ref").ThrowAsJavaScriptException();
        return;
  }
  auto wrapper_ptr_value = info[0].As<Napi::External<xla::XlaOp>>();
  if (!wrapper_ptr_value.CheckTypeTag(&type_tag_)) {
        Napi::Error::New(env, "XlaOp can only be created with external ref").ThrowAsJavaScriptException();
        return;
  }

  op_.reset(wrapper_ptr_value.Data());
}

Napi::Value XlaOpWrapper::Dump(const Napi::CallbackInfo& info) {
  std::cout << *(op_.get()) << "\n";
  return info.Env().Null();
}

class XlaComputationWrapper: public Napi::ObjectWrap<XlaComputationWrapper> {
  public:
    static Napi::Object Init(Napi::Env env, Napi::Object exports);
    XlaComputationWrapper(const Napi::CallbackInfo& info);

    Napi::Value Dump(const Napi::CallbackInfo& info);

    static Napi::Object create(Napi::Env env, xla::XlaComputation *op);
  private:
    static Napi::FunctionReference constructor;
    static napi_type_tag type_tag_;

    std::unique_ptr<xla::XlaComputation> op_;
};

napi_type_tag XlaComputationWrapper::type_tag_ = { 0xdf6e6fbb6c724f60ULL, 0x842a52de6b318692ULL };

Napi::FunctionReference XlaComputationWrapper::constructor;

// static 
Napi::Object XlaComputationWrapper::create(Napi::Env env, xla::XlaComputation *op) {
  auto r = Napi::External<xla::XlaComputation>::New(env, op);
  r.TypeTag(&type_tag_);
  return XlaComputationWrapper::constructor.New({r});
}

Napi::Object XlaComputationWrapper::Init(Napi::Env env, Napi::Object exports) {
  Napi::Function func = DefineClass(
      env, "XlaComputation",
      {});

  constructor = Napi::Persistent(func);
  constructor.SuppressDestruct();

  exports.Set("XlaComputation", func);
  return exports;
}

XlaComputationWrapper::XlaComputationWrapper(const Napi::CallbackInfo& info) : Napi::ObjectWrap<XlaComputationWrapper>(info) {
  Napi::Env env = info.Env();
  if (info.Length() != 1 || !info[0].IsExternal()) {
        Napi::Error::New(env, "XlaComputation can only be created with external ref").ThrowAsJavaScriptException();
        return;
  }
  auto wrapper_ptr_value = info[0].As<Napi::External<xla::XlaComputation>>();
  if (!wrapper_ptr_value.CheckTypeTag(&type_tag_)) {
        Napi::Error::New(env, "XlaComputation can only be created with external ref").ThrowAsJavaScriptException();
        return;
  }

  op_.reset(wrapper_ptr_value.Data());
}


static absl::StatusOr<float> example() {
  ASSIGN_OR_RETURN(client, xla::GetTfrtCpuClient(false));

  printf("X: %i\n", __LINE__);

  xla::XlaBuilder *builder = new xla::XlaBuilder("fn");
  printf("X: %i\n", __LINE__);
  xla::XlaOp op = xla::ConstantR1<float>(builder, {42.1f, 42.4f});
  printf("X: %i\n", __LINE__);

  ASSIGN_OR_RETURN(executable, builder->Build(&op));
  printf("X: %i\n", __LINE__);

  ASSIGN_OR_RETURN(loaded_executable,
                   client->Compile(executable, xla::CompileOptions{}));
  printf("X: %i\n", __LINE__);

  // auto local_literal = xla::LiteralUtil::CreateR0<float>(3.0f);
  // ASSIGN_OR_RETURN(arg, client->BufferFromHostLiteral(local_literal,
  // client->devices()[0]));

  ASSIGN_OR_RETURN(results, loaded_executable->Execute({{}}, {}));
  printf("X: %i\n", __LINE__);

  ASSIGN_OR_RETURN(out, results[0][0]->ToLiteralSync());

  printf("X: %i\n", __LINE__);
  std::cout << out->shape().ToString()
            << " (dims: " << out->shape().dimensions_size() << ")\n";
  printf("X: %i\n", __LINE__);

  return out->data<float>()[0];
}

Napi::Value ConstantR0F32(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() != 2 || !info[0].IsObject() || !info[1].IsNumber()) {
        Napi::Error::New(env, "ConstantR0 expects a builder and a number").ThrowAsJavaScriptException();
        return env.Null();
  }

  auto builder = XlaBuilderWrapper::BuilderFromValue(env, info[0]);
  if (!builder) return env.Null();
  xla::XlaOp *op = new xla::XlaOp(xla::ConstantR0<float>(builder, info[1].As<Napi::Number>()));
  return XlaOpWrapper::create(env, op);
}

Napi::Object Init(Napi::Env env, Napi::Object exports) {
  Napi::HandleScope scope(env);

  // example();

  XlaClient::Init(env, exports);
  XlaBuilderWrapper::Init(env, exports);
  XlaOpWrapper::Init(env, exports);
  XlaComputationWrapper::Init(env, exports);

  exports.Set(Napi::String::New(env, "constantR0f32"), Napi::Function::New<ConstantR0F32>(env));

  printf("Hello!?\n");
  return exports;
}

NODE_API_MODULE(testing, Init);