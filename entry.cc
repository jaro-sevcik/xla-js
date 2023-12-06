#include <xla_js.h>

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

Napi::Value XlaClient::Compile(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  printf("Compiling!");
  return env.Null();
}

XlaClient::XlaClient(const Napi::CallbackInfo &info)
    : Napi::ObjectWrap<XlaClient>(info) {
  Napi::Env env = info.Env();

  auto client_status = xla::GetTfrtCpuClient(false);
  if (!client_status.ok()) {
    Napi::Error::New(env, "XLA client initialization failed")
        .ThrowAsJavaScriptException();
    return;
  }

  this->client_ = std::move(client_status.value());
}

Napi::Object XlaClient::Init(Napi::Env env, Napi::Object exports) {
  Napi::Function func =
      DefineClass(env, "Client",
                  {
                      InstanceMethod("compile", &XlaClient::Compile,
                                     static_cast<napi_property_attributes>(
                                         napi_writable | napi_configurable)),
                  });

  exports.Set("Client", func);
  return exports;
}
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

template <class T, typename TPtr>
Napi::FunctionReference ReferenceWrapper<T, TPtr>::constructor = {};

// static
template <class T, typename TPtr>
Napi::Object ReferenceWrapper<T, TPtr>::Init(
    Napi::Env env, Napi::Object exports, const char *name,
    const std::initializer_list<Napi::ClassPropertyDescriptor<T>> &properties) {

  Napi::Function func =
      ReferenceWrapper<T, TPtr>::DefineClass(env, name, properties);

  constructor = Napi::Persistent(func);
  constructor.SuppressDestruct();

  exports.Set(name, func);
  return exports;
}

template <class T, typename TPtr>
Napi::Object ReferenceWrapper<T, TPtr>::Create(Napi::Env env, TPtr *op) {
  auto r = Napi::External<TPtr>::New(env, op);
  r.TypeTag(&type_tag_);
  return ReferenceWrapper::constructor.New({r});
}

// static
template <class T, typename TPtr>
TPtr *ReferenceWrapper<T, TPtr>::FromValue(Napi::Env env, Napi::Value value) {
  if (!value.IsObject() || !value.As<Napi::Object>().CheckTypeTag(&type_tag_)) {
    Napi::Error::New(env, "Invalid argument").ThrowAsJavaScriptException();
    return nullptr;
  }

  return ReferenceWrapper<T, TPtr>::Unwrap(value.As<Napi::Object>())
      ->value_.get();
}

template <class T, typename TPtr>
ReferenceWrapper<T, TPtr>::ReferenceWrapper(const Napi::CallbackInfo &info,
                                            TPtr *ptr)
    : Napi::ObjectWrap<T>(info) {
  info.This().As<Napi::Object>().TypeTag(&type_tag_);
  if (ptr) {
    value_.reset(ptr);
    return;
  }

  Napi::Env env = info.Env();
  if (env.IsExceptionPending())
    return;
  if (info.Length() != 1 || !info[0].IsExternal()) {
    Napi::Error::New(env, "Wrapper can only be created with external ref")
        .ThrowAsJavaScriptException();
    return;
  }
  auto wrapper_ptr_value = info[0].As<Napi::External<TPtr>>();
  if (!wrapper_ptr_value.CheckTypeTag(&type_tag_)) {
    Napi::Error::New(env, "Wrapper can only be created with external ref")
        .ThrowAsJavaScriptException();
    return;
  }

  value_.reset(wrapper_ptr_value.Data());
}

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

template <>
napi_type_tag ReferenceWrapper<XlaOpWrapper, xla::XlaOp>::type_tag_ = {
    0xdf6e6fbb6c724f60ULL, 0x842a52de6b318692ULL};

Napi::Object XlaOpWrapper::Init(Napi::Env env, Napi::Object exports) {
  return ReferenceWrapper<XlaOpWrapper, xla::XlaOp>::Init(
      env, exports, "XlaOp",
      {
          InstanceMethod("dump", &XlaOpWrapper::Dump,
                         static_cast<napi_property_attributes>(
                             napi_writable | napi_configurable)),
      });
}

Napi::Value XlaOpWrapper::Dump(const Napi::CallbackInfo &info) {
  std::cout << *(value_.get()) << "\n";
  return info.Env().Null();
}

class XlaComputationWrapper
    : public ReferenceWrapper<XlaComputationWrapper, xla::XlaComputation> {
public:
  static Napi::Object Init(Napi::Env env, Napi::Object exports);
  XlaComputationWrapper(const Napi::CallbackInfo &info)
      : ReferenceWrapper<XlaComputationWrapper, xla::XlaComputation>(info) {}
};

// XlaComputation wrapper
template <>
napi_type_tag
    ReferenceWrapper<XlaComputationWrapper, xla::XlaComputation>::type_tag_ = {
        0xdf6e6fbb6c724f60ULL, 0x842a52de6b318692ULL};

Napi::Object XlaComputationWrapper::Init(Napi::Env env, Napi::Object exports) {
  return ReferenceWrapper<XlaComputationWrapper, xla::XlaComputation>::Init(
      env, exports, "XlaComputation", {});
}

class LoadedExecutableWrapper
    : public ReferenceWrapper<LoadedExecutableWrapper,
                              xla::PjRtLoadedExecutable> {
public:
  static Napi::Object Init(Napi::Env env, Napi::Object exports);
  LoadedExecutableWrapper(const Napi::CallbackInfo &info)
      : ReferenceWrapper<LoadedExecutableWrapper, xla::PjRtLoadedExecutable>(
            info) {}
};

// LoadedExecutable wrapper
template <>
napi_type_tag ReferenceWrapper<LoadedExecutableWrapper,
                               xla::PjRtLoadedExecutable>::type_tag_ = {
    0xdf6e6fbb6c724f60ULL, 0x842a52de6b318693ULL};

Napi::Object LoadedExecutableWrapper::Init(Napi::Env env,
                                           Napi::Object exports) {
  return ReferenceWrapper<LoadedExecutableWrapper,
                          xla::PjRtLoadedExecutable>::Init(env, exports,
                                                           "LoadedExecutable",
                                                           {});
}

// XlaBuilder wrapper
template <>
napi_type_tag ReferenceWrapper<XlaBuilderWrapper, xla::XlaBuilder>::type_tag_ =
    {0xdf6e6fbb6c724f60ULL, 0x842a52de6b318691ULL};

// static
xla::XlaBuilder *
XlaBuilderWrapper::createBuilder(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if (info.Length() != 1 || !info[0].IsString()) {
    Napi::Error::New(
        env,
        "XlaBuilder constructor expects string name as its single argument")
        .ThrowAsJavaScriptException();
    return nullptr;
  }

  auto str = info[0].As<Napi::String>().Utf8Value();

  return new xla::XlaBuilder(str.c_str());
}

Napi::Object XlaBuilderWrapper::Init(Napi::Env env, Napi::Object exports) {
  return ReferenceWrapper<XlaBuilderWrapper, xla::XlaBuilder>::Init(
      env, exports, "XlaBuilder",
      {
          InstanceMethod("build", &XlaBuilderWrapper::Build,
                         static_cast<napi_property_attributes>(
                             napi_writable | napi_configurable)),
      });
}

Napi::Value XlaBuilderWrapper::Build(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if (info.Length() != 1) {
    Napi::Error::New(env,
                     "XlaBuilder.Build expects XlaOp as its single argument")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  xla::XlaOp *op = XlaOpWrapper::FromValue(env, info[1]);
  if (env.IsExceptionPending())
    return env.Null();

  auto computation_or_status = this->value()->Build(op);
  if (!computation_or_status.ok()) {
    Napi::Error::New(env, computation_or_status.status().ToString().c_str())
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  xla::XlaComputation *computation =
      new xla::XlaComputation(std::move(computation_or_status.value()));

  return XlaComputationWrapper::Create(env, computation);
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

Napi::Value ConstantR0F32(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if (info.Length() != 2 || !info[0].IsObject() || !info[1].IsNumber()) {
    Napi::Error::New(env, "ConstantR0 expects a builder and a number")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto builder = XlaBuilderWrapper::FromValue(env, info[0]);
  if (!builder)
    return env.Null();
  xla::XlaOp *op = new xla::XlaOp(
      xla::ConstantR0<float>(builder, info[1].As<Napi::Number>()));
  return XlaOpWrapper::Create(env, op);
}

Napi::Object Init(Napi::Env env, Napi::Object exports) {
  Napi::HandleScope scope(env);

  // example();

  XlaClient::Init(env, exports);
  XlaBuilderWrapper::Init(env, exports);
  XlaOpWrapper::Init(env, exports);
  XlaComputationWrapper::Init(env, exports);

  exports.Set(Napi::String::New(env, "constantR0f32"),
              Napi::Function::New<ConstantR0F32>(env));

  printf("Hello!?\n");
  return exports;
}

NODE_API_MODULE(testing, Init);