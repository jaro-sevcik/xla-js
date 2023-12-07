#include <xla_js.h>

#define CONCAT(a, b) a##b

#define ASSIGN_OR_RETURN(name, rhs)                                            \
  ASSIGN_OR_RETURN_WITH_TEMP(CONCAT(name, __COUNTER__), name, rhs)

#define ASSIGN_OR_RETURN_WITH_TEMP(temp, name, rhs)                            \
  auto temp = rhs;                                                             \
  if (!temp.ok())                                                              \
    return temp.status();                                                      \
  auto name = std::move(temp.value());

#define ASSIGN_OR_THROW(name, rhs)                                             \
  ASSIGN_OR_THROW_WITH_TEMP(                                                   \
      CONCAT(name, __COUNTER__), name, rhs,                                    \
      CONCAT(name, __COUNTER__).status().ToString().c_str(), env.Null())

#define ASSIGN_OR_THROW_RETURN(name, rhs)                                      \
  ASSIGN_OR_THROW_WITH_TEMP(                                                   \
      CONCAT(name, __COUNTER__), name, rhs,                                    \
      CONCAT(name, __COUNTER__).status().ToString().c_str(),                   \
      /**/)

#define ASSIGN_OR_THROW_MESSAGE(name, rhs, message)                            \
  ASSIGN_OR_THROW_WITH_TEMP(CONCAT(name, __COUNTER__), name, rhs, message,     \
                            env.Null())

#define ASSIGN_OR_THROW_WITH_TEMP(temp, name, rhs, message, retval)            \
  auto temp = (rhs);                                                           \
  if (!temp.ok()) {                                                            \
    Napi::Error::New(env, message).ThrowAsJavaScriptException();               \
    return retval;                                                             \
  }                                                                            \
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

Napi::Value PjRtClient::Compile(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  const char *msg = "Client.build expects executable and options as arguments";

  if (info.Length() != 2 || !info[0].IsObject() || !info[1].IsObject()) {
    Napi::Error::New(env, msg).ThrowAsJavaScriptException();
    return env.Null();
  }

  auto computation = XlaComputationWrapper::FromValue(env, info[0]);

  ASSIGN_OR_THROW(loaded_executable,
                  client_->Compile(*computation, xla::CompileOptions{}));

  printf("Compiling!\n");

  auto le = loaded_executable.release();
  return PjRtLoadedExecutableWrapper::Create(env, le);
}

PjRtClient::PjRtClient(const Napi::CallbackInfo &info)
    : Napi::ObjectWrap<PjRtClient>(info) {
  Napi::Env env = info.Env();

  ASSIGN_OR_THROW_RETURN(client, xla::GetTfrtCpuClient(false));
  this->client_ = std::move(client);
}

Napi::Object PjRtClient::Init(Napi::Env env, Napi::Object exports) {
  Napi::Function func =
      DefineClass(env, "Client",
                  {
                      InstanceMethod("compile", &PjRtClient::Compile,
                                     static_cast<napi_property_attributes>(
                                         napi_writable | napi_configurable)),
                  });

  exports.Set("Client", func);
  return exports;
}

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
  return ReferenceWrapper<T, TPtr>::constructor.New({r});
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

// XlaComputation wrapper
template <>
napi_type_tag
    ReferenceWrapper<XlaComputationWrapper, xla::XlaComputation>::type_tag_ = {
        0xdf6e6fbb6c724f60ULL, 0x842a52de6b318692ULL};

Napi::Object XlaComputationWrapper::Init(Napi::Env env, Napi::Object exports) {
  return ReferenceWrapper<XlaComputationWrapper, xla::XlaComputation>::Init(
      env, exports, "XlaComputation", {});
}

// PjRtLoadedExecutable wrapper
template <>
napi_type_tag ReferenceWrapper<PjRtLoadedExecutableWrapper,
                               xla::PjRtLoadedExecutable>::type_tag_ = {
    0xdf6e6fbb6c724f60ULL, 0x842a52de6b318693ULL};

Napi::Object PjRtLoadedExecutableWrapper::Init(Napi::Env env,
                                               Napi::Object exports) {
  return ReferenceWrapper<PjRtLoadedExecutableWrapper,
                          xla::PjRtLoadedExecutable>::
      Init(env, exports, "PjRtLoadedExecutable",
           {InstanceMethod("execute", &PjRtLoadedExecutableWrapper::Execute,
                           static_cast<napi_property_attributes>(
                               napi_writable | napi_configurable))});
}

Napi::Value
PjRtLoadedExecutableWrapper::Execute(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();

  // TODO thread through the inputs!
  ASSIGN_OR_THROW(results, value()->Execute({{}}, {}));

  auto result_array = Napi::Array::New(env, results.size());
  for (size_t i = 0; i < results.size(); i++) {
    auto &outer = results[i];
    auto inner_array = Napi::Array::New(env, outer.size());
    for (size_t j = 0; j < outer.size(); j++) {
      auto &inner = outer[j];
      inner_array.Set(j, PjRtBufferWrapper::Create(env, inner.release()));
    }
    result_array.Set(i, inner_array);
  }
  return result_array;
}

// PjRtBuffer wrapper
template <>
napi_type_tag ReferenceWrapper<PjRtBufferWrapper, xla::PjRtBuffer>::type_tag_ =
    {0xdf6e6fbb6c724f60ULL, 0x842a52de6b318694ULL};

Napi::Object PjRtBufferWrapper::Init(Napi::Env env, Napi::Object exports) {
  return ReferenceWrapper<PjRtBufferWrapper, xla::PjRtBuffer>::Init(
      env, exports, "PjRtBuffer",
      {
          InstanceMethod("toLiteralSync", &PjRtBufferWrapper::ToLiteralSync,
                         static_cast<napi_property_attributes>(
                             napi_writable | napi_configurable)),
      });
}

Napi::Value PjRtBufferWrapper::ToLiteralSync(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  ASSIGN_OR_THROW(literal_sp, value()->ToLiteralSync());

  xla::Literal *literal = new xla::Literal(std::move(*literal_sp));

  return LiteralWrapper::Create(env, literal);
}

// Literal wrapper
template <>
napi_type_tag ReferenceWrapper<LiteralWrapper, xla::Literal>::type_tag_ = {
    0xdf6e6fbb6c724f60ULL, 0x842a52de6b318695ULL};

Napi::Object LiteralWrapper::Init(Napi::Env env, Napi::Object exports) {
  return ReferenceWrapper<LiteralWrapper, xla::Literal>::Init(
      env, exports, "Literal",
      {
          InstanceMethod("getFirstElementF32",
                         &LiteralWrapper::GetFirstElementF32,
                         static_cast<napi_property_attributes>(
                             napi_writable | napi_configurable)),

      });
}

Napi::Value LiteralWrapper::GetFirstElementF32(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  float result = value()->GetFirstElement<float>();
  return Napi::Number::New(env, result);
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
                     "XlaBuilder.build expects XlaOp as its single argument")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  xla::XlaOp *op = XlaOpWrapper::FromValue(env, info[0]);
  if (env.IsExceptionPending())
    return env.Null();

  ASSIGN_OR_THROW(computation_instance, this->value()->Build(op));

  return XlaComputationWrapper::Create(
      env, new xla::XlaComputation(std::move(computation_instance)));
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

  PjRtClient::Init(env, exports);
  XlaBuilderWrapper::Init(env, exports);
  XlaOpWrapper::Init(env, exports);
  XlaComputationWrapper::Init(env, exports);
  PjRtLoadedExecutableWrapper::Init(env, exports);
  PjRtBufferWrapper::Init(env, exports);
  LiteralWrapper::Init(env, exports);

  exports.Set(Napi::String::New(env, "constantR0f32"),
              Napi::Function::New<ConstantR0F32>(env));

  printf("Hello!?\n");
  return exports;
}

NODE_API_MODULE(testing, Init);