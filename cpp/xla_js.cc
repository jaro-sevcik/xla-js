#include "xla_js.h"
#include <sstream>
#include <tuple>
#include <vector>

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
  if (env.IsExceptionPending()) {                                              \
    return retval;                                                             \
  }                                                                            \
  auto temp = (rhs);                                                           \
  if (!temp.ok()) {                                                            \
    Napi::Error::New(env, message).ThrowAsJavaScriptException();               \
    return retval;                                                             \
  }                                                                            \
  auto name = std::move(temp.value());

namespace {

std::vector<float> floatArrayFromArray(Napi::Env env, Napi::Array array) {
  std::vector<float> nums;
  for (size_t i = 0; i < array.Length(); i++) {
    Napi::Value e = array[i];
    if (!e.IsNumber()) {
      Napi::Error::New(
          env, "createR1 requires an array of numbers as the second argument")
          .ThrowAsJavaScriptException();
      return {};
    }
    nums.push_back(e.As<Napi::Number>().FloatValue());
  }
  return nums;
}

std::vector<int64_t> int64ArrayFromNapiArray(Napi::Env env, Napi::Array array) {
  std::vector<int64_t> dims;
  if (env.IsExceptionPending())
    return {};
  for (size_t i = 0; i < array.Length(); i++) {
    Napi::Value e = array[i];
    if (!e.IsNumber()) {
      Napi::Error::New(env, "Number elements expected")
          .ThrowAsJavaScriptException();
      return {};
    }
    dims.push_back(e.As<Napi::Number>().Int64Value());
    if (env.IsExceptionPending())
      return {};
  }
  return dims;
}

template <typename H, typename T> struct TupleCons;

template <typename H, typename... Ts> struct TupleCons<H, std::tuple<Ts...>> {
  using type = std::tuple<H, Ts...>;
};

template <typename T> struct ElementConverter;

template <> struct ElementConverter<int32_t> {
  using Result = int32_t;

  static Result convert(Napi::Env env, Napi::Value value) {
    if (env.IsExceptionPending())
      return {};
    return value.As<Napi::Number>().Int32Value();
  }
};

template <> struct ElementConverter<int64_t> {
  using Result = int64_t;

  static Result convert(Napi::Env env, Napi::Value value) {
    if (env.IsExceptionPending())
      return {};
    return value.As<Napi::Number>().Int64Value();
  }
};

template <> struct ElementConverter<std::string> {
  using Result = std::string;

  static Result convert(Napi::Env env, Napi::Value value) {
    if (env.IsExceptionPending())
      return {};
    return value.As<Napi::String>().Utf8Value();
  }
};

template <> struct ElementConverter<std::vector<int64_t>> {
  using Result = std::vector<int64_t>;

  static Result convert(Napi::Env env, Napi::Value value) {
    if (env.IsExceptionPending())
      return {};
    return int64ArrayFromNapiArray(env, value.As<Napi::Array>());
  }
};

template <> struct ElementConverter<std::vector<xla::XlaOp>> {
  using Result = std::vector<xla::XlaOp>;

  static Result convert(Napi::Env env, Napi::Value value) {
    if (env.IsExceptionPending())
      return {};

    Result result;

    if (value.IsArray()) {
      Napi::Array array = value.As<Napi::Array>();
      for (size_t i = 0; i < array.Length(); i++) {
        Napi::Value e = array[i];
        result.push_back(xla::XlaOp(*XlaOpWrapper::FromValue(env, e)));
        if (env.IsExceptionPending())
          return {};
      }
    } else {
      result.push_back(xla::XlaOp(*XlaOpWrapper::FromValue(env, value)));
      if (env.IsExceptionPending())
        return {};
    }

    return result;
  }
};

template <> struct ElementConverter<xla::XlaBuilder *> {
  using Result = xla::XlaBuilder *;

  static Result convert(Napi::Env env, Napi::Value value) {
    if (env.IsExceptionPending())
      return {};
    return XlaBuilderWrapper::FromValue(env, value);
  }
};

template <> struct ElementConverter<xla::XlaOp *> {
  using Result = xla::XlaOp *;

  static Result convert(Napi::Env env, Napi::Value value) {
    if (env.IsExceptionPending())
      return {};
    return XlaOpWrapper::FromValue(env, value);
  }
};

template <> struct ElementConverter<xla::PjRtBuffer *> {
  using Result = xla::PjRtBuffer *;

  static Result convert(Napi::Env env, Napi::Value value) {
    if (env.IsExceptionPending())
      return {};
    return PjRtBufferWrapper::FromValue(env, value);
  }
};

template <> struct ElementConverter<xla::Shape *> {
  using Result = xla::Shape *;

  static Result convert(Napi::Env env, Napi::Value value) {
    if (env.IsExceptionPending())
      return {};
    return ShapeWrapper::FromValue(env, value);
  }
};

template <> struct ElementConverter<xla::XlaComputation *> {
  using Result = xla::XlaComputation *;

  static Result convert(Napi::Env env, Napi::Value value) {
    if (env.IsExceptionPending())
      return {};
    return XlaComputationWrapper::FromValue(env, value);
  }
};

template <> struct ElementConverter<xla::Literal *> {
  using Result = xla::Literal *;

  static Result convert(Napi::Env env, Napi::Value value) {
    if (env.IsExceptionPending())
      return {};
    return LiteralWrapper::FromValue(env, value);
  }
};

// template<>
// struct ElementConverter<double> {
//     using Result = std::string;

//     static Result convert(int p) {
//         std::ostringstream ss;
//         ss << "Ha, double (" << p << ")";
//         return ss.str();
//     }
// };

template <size_t index, typename... Ts> class ListConverter;

template <size_t index> class ListConverter<index> {
public:
  using Result = std::tuple<>;

  static Result convert(const Napi::CallbackInfo &info) {
    return std::make_tuple();
  }
};

template <size_t index, typename T, typename... Rest>
class ListConverter<index, T, Rest...> {
public:
  using Conv = ElementConverter<T>;
  using RestConv = ListConverter<index + 1, Rest...>;
  using Result = typename TupleCons<typename Conv::Result,
                                    typename RestConv::Result>::type;

  static Result convert(const Napi::CallbackInfo &info) {
    Napi::Env env = info.Env();
    typename Conv::Result r = Conv::convert(env, info[index]);
    return std::tuple_cat(std::make_tuple(r), RestConv::convert(info));
  }
};

template <typename... Ts>
typename ListConverter<0, Ts...>::Result match(const Napi::CallbackInfo &info,
                                               const char *name) {
  if (sizeof...(Ts) != info.Length()) {
    Napi::Env env = info.Env();
    std::string msg = absl::StrFormat(
        "Method %s takes %zi arguments, instead received %zi arguments.", name,
        sizeof...(Ts), info.Length());
    Napi::Error::New(env, msg.c_str()).ThrowAsJavaScriptException();
  }
  return ListConverter<0, Ts...>::convert(info);
}

} // namespace

// PjRtClient
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

  auto le = loaded_executable.release();
  return PjRtLoadedExecutableWrapper::Create(env, le);
}

Napi::Value PjRtClient::BufferFromHostLiteral(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  const char *msg = "bufferFromHostLiteral requires a literal argument";
  if (info.Length() < 1 || !info[0].IsObject() || info.Length() > 2 ||
      (info.Length() == 2 && !info[1].IsNumber())) {
    Napi::Error::New(env, msg).ThrowAsJavaScriptException();
    return env.Null();
  }

  size_t device_id = 0;
  if (info.Length() == 2) {
    device_id = info[1].As<Napi::Number>().Uint32Value();
  }

  auto literal = LiteralWrapper::FromValue(env, info[0]);
  auto device = value()->devices()[device_id];
  ASSIGN_OR_THROW(buffer, value()->BufferFromHostLiteral(*literal, device));

  return PjRtBufferWrapper::Create(env, buffer.release());
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
                      InstanceMethod("bufferFromHostLiteral",
                                     &PjRtClient::BufferFromHostLiteral,
                                     static_cast<napi_property_attributes>(
                                         napi_writable | napi_configurable)),
                  });

  exports.Set("Client", func);
  return exports;
}

// ReferenceWrapper
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
  const char *msg =
      "PjRtLoadedExecutableWrapper.execute expects "
      "array of arrays of buffers executable and options as arguments";

  if (info.Length() != 2 || !info[0].IsArray() || !info[1].IsObject()) {
    Napi::Error::New(env, msg).ThrowAsJavaScriptException();
    return env.Null();
  }

  std::vector<std::vector<xla::PjRtBuffer *>> exec_inputs;

  auto inputs_arg = info[0].As<Napi::Array>();

  for (size_t i = 0; i < inputs_arg.Length(); i++) {
    Napi::Value input_list_val = inputs_arg[i];
    if (!input_list_val.IsArray()) {
      Napi::Error::New(env, msg).ThrowAsJavaScriptException();
      return env.Null();
    }
    auto input_list = input_list_val.As<Napi::Array>();
    std::vector<xla::PjRtBuffer *> inputs_exec_elem;
    for (size_t j = 0; j < input_list.Length(); j++) {
      auto buffer = PjRtBufferWrapper::FromValue(env, input_list[j]);
      inputs_exec_elem.push_back(buffer);
    }
    exec_inputs.push_back(std::move(inputs_exec_elem));
  }

  // TODO thread through the inputs!
  ASSIGN_OR_THROW(results, value()->Execute(exec_inputs, {}));

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
          InstanceMethod("get", &LiteralWrapper::Get,
                         static_cast<napi_property_attributes>(
                             napi_writable | napi_configurable)),
          InstanceMethod("data", &LiteralWrapper::Data,
                         static_cast<napi_property_attributes>(
                             napi_writable | napi_configurable)),
          InstanceMethod("shape", &LiteralWrapper::Shape,
                         static_cast<napi_property_attributes>(
                             napi_writable | napi_configurable)),
          InstanceMethod("reshape", &LiteralWrapper::Reshape,
                         static_cast<napi_property_attributes>(
                             napi_writable | napi_configurable)),
          InstanceMethod("broadcast", &LiteralWrapper::Broadcast,
                         static_cast<napi_property_attributes>(
                             napi_writable | napi_configurable)),
          InstanceMethod("toString", &LiteralWrapper::ToString,
                         static_cast<napi_property_attributes>(
                             napi_writable | napi_configurable)),
          StaticMethod("createR0", &LiteralWrapper::CreateR0,
                       static_cast<napi_property_attributes>(
                           napi_writable | napi_configurable)),
          StaticMethod("createR1", &LiteralWrapper::CreateR1,
                       static_cast<napi_property_attributes>(
                           napi_writable | napi_configurable)),
      });
}

Napi::Value LiteralWrapper::GetFirstElementF32(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  float result = value()->GetFirstElement<float>();
  return Napi::Number::New(env, result);
}

Napi::Value LiteralWrapper::Get(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();

  auto [indices] = match<std::vector<int64_t>>(info, "Literal.get");
  if (env.IsExceptionPending())
    return env.Null();

  if (value()->shape().rank() != indices.size()) {
    Napi::Error::New(env, "index length does not match dimensions")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  for (size_t i = 0; i < value()->shape().rank(); i++) {
    if (value()->shape().dimensions(i) <= indices[i]) {
      std::string msg = absl::StrFormat(
          "Index %lli at dimension %zi is out of bounds (size is %lli)",
          indices[i], i, value()->shape().dimensions(i));
      Napi::Error::New(env, msg.c_str()).ThrowAsJavaScriptException();
      return env.Null();
    }
  }

  switch (value()->shape().element_type()) {
  case xla::F32:
    return Napi::Number::New(env, value()->Get<float>(absl::Span<const int64_t>(
                                      indices.data(), indices.size())));
  default:
    break;
  }

  std::string msg =
      absl::StrFormat("Get for element type '%s' is not supported",
                      xla::PrimitiveType_Name(value()->shape().element_type()));
  Napi::Error::New(env, msg.c_str()).ThrowAsJavaScriptException();
  return env.Null();
}

namespace {

template <typename T>
Napi::Array napiArrayFromXlaArray(Napi::Env env, absl::Span<T> data) {
  Napi::Array result = Napi::Array::New(env, data.size());
  for (size_t i = 0; i < data.size(); i++) {
    result[i] = Napi::Number::New(env, data[i]);
  }
  return result;
}

} // namespace

Napi::Value LiteralWrapper::Data(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if (info.Length() != 1 || !info[0].IsString()) {
    Napi::Error::New(
        env,
        "Data requires a primitive type and a number constant as arguments")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  std::string ptype = info[0].As<Napi::String>().Utf8Value();

  if (ptype == "F32") {
    auto data = value()->data<float>();
    return napiArrayFromXlaArray<const float>(env, data);
  }

  Napi::Error::New(env, "data(): unsupported primitive type")
      .ThrowAsJavaScriptException();
  return env.Null();
}

Napi::Value LiteralWrapper::Shape(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  auto shape = new xla::Shape(value()->shape());
  return ShapeWrapper::Create(env, shape);
}

Napi::Value LiteralWrapper::Reshape(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  auto [dims] = match<std::vector<int64_t>>(info, "Literal.reshape");
  if (env.IsExceptionPending())
    return env.Null();

  ASSIGN_OR_THROW(reshaped, value()->Reshape(absl::Span<const int64_t>(
                                dims.data(), dims.size())));

  return LiteralWrapper::Create(env, new xla::Literal(std::move(reshaped)));
}

Napi::Value LiteralWrapper::Broadcast(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  auto [shape, dims] =
      match<xla::Shape *, std::vector<int64_t>>(info, "Literal.broadcast");
  if (env.IsExceptionPending())
    return env.Null();
  ASSIGN_OR_THROW(broadcasted,
                  value()->Broadcast(*shape, absl::Span<const int64_t>(
                                                 dims.data(), dims.size())));
  return LiteralWrapper::Create(env, new xla::Literal(std::move(broadcasted)));
}

Napi::Value LiteralWrapper::ToString(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  return Napi::String::New(env, value()->ToString().c_str());
}

// static
Napi::Value LiteralWrapper::CreateR0(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if (info.Length() != 2 || !info[0].IsString() || !info[1].IsNumber()) {
    Napi::Error::New(
        env,
        "createR0 requires a primitive type and a number constant as arguments")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  std::string ptype = info[0].As<Napi::String>().Utf8Value();

  xla::Literal *literal = nullptr;
  if (ptype == "F32") {
    literal = new xla::Literal(
        xla::LiteralUtil::CreateR0<float>(info[1].As<Napi::Number>()));
  }

  if (!literal) {
    Napi::Error::New(env, "createR0: unsupported primitive type")
        .ThrowAsJavaScriptException();
  }

  return Create(env, literal);
}

Napi::Value LiteralWrapper::CreateR1(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if (info.Length() != 2 || !info[0].IsString() || !info[1].IsArray()) {
    Napi::Error::New(env, "createR1 requires a primitive type and an array of "
                          "numbers as arguments")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  std::string ptype = info[0].As<Napi::String>().Utf8Value();
  auto array = info[1].As<Napi::Array>();

  xla::Literal *literal = nullptr;
  if (ptype == "F32") {
    std::vector<float> nums = floatArrayFromArray(env, array);
    if (env.IsExceptionPending())
      return env.Null();
    literal = new xla::Literal(xla::LiteralUtil::CreateR1<float>(
        absl::Span<const float>(nums.data(), nums.size())));
  }

  if (!literal) {
    Napi::Error::New(env, "createR1: unsupported primitive type")
        .ThrowAsJavaScriptException();
  }

  return Create(env, literal);
}

// Shape wrapper
template <>
napi_type_tag ReferenceWrapper<ShapeWrapper, xla::Shape>::type_tag_ = {
    0xdf6e6fbb6c724f60ULL, 0x842a52de6b318696ULL};

Napi::Object ShapeWrapper::Init(Napi::Env env, Napi::Object exports) {
  return ReferenceWrapper<ShapeWrapper, xla::Shape>::Init(
      env, exports, "Shape",
      {
          InstanceMethod("dimensions", &ShapeWrapper::Dimensions,
                         static_cast<napi_property_attributes>(
                             napi_writable | napi_configurable)),
          InstanceMethod("primitiveType", &ShapeWrapper::PrimitiveType,
                         static_cast<napi_property_attributes>(
                             napi_writable | napi_configurable)),
          StaticMethod("forArray", &ShapeWrapper::ForArray,
                       static_cast<napi_property_attributes>(
                           napi_writable | napi_configurable)),

      });
}

Napi::Value ShapeWrapper::Dimensions(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if (!value()->IsArray()) {
    return Napi::Array::New(env, 0);
  }

  int dims = value()->dimensions_size();
  auto result = Napi::Array::New(env, value()->dimensions_size());
  for (int i = 0; i < dims; i++) {
    result[i] = value()->dimensions(i);
  }
  return result;
}

Napi::Value ShapeWrapper::PrimitiveType(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();

  const char *p = nullptr;
  switch (value()->element_type()) {
#define PRIMITIVE_TYPE_CASE(P)                                                 \
  case xla::P:                                                                 \
    p = #P;                                                                    \
    break;
    PRIMITIVE_TYPE_V(PRIMITIVE_TYPE_CASE)
#undef PRIMITIVE_TYPE_CASE
  default:
    break;
  }

  if (p) {
    return Napi::String::New(env, p);
  }

  return env.Null();
}

// static
Napi::Value ShapeWrapper::ForArray(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  const char *msg = "Shape.forArray expects a primitive type and "
                    "an array of dimension sizes";

  if (info.Length() != 2 || !info[0].IsString() || !info[1].IsArray()) {
    Napi::Error::New(env, msg).ThrowAsJavaScriptException();
    return env.Null();
  }

  auto p = info[0].As<Napi::String>().Utf8Value();
  xla::PrimitiveType primitive_type = xla::PRIMITIVE_TYPE_INVALID;
#define PRIMITIVE_TYPE_CASE(P)                                                 \
  if (p == #P)                                                                 \
    primitive_type = xla::P;
  PRIMITIVE_TYPE_V(PRIMITIVE_TYPE_CASE)
#undef PRIMITIVE_TYPE_CASE

  if (primitive_type == xla::PRIMITIVE_TYPE_INVALID) {
    Napi::Error::New(env, msg).ThrowAsJavaScriptException();
    return env.Null();
  }

  auto dim_args = info[1].As<Napi::Array>();
  std::vector<int64_t> dims;
  std::vector<bool> dynamic_dims;
  for (size_t i = 0; i < dim_args.Length(); i++) {
    Napi::Value dim_arg = dim_args[i];
    if (!dim_arg.IsNumber()) {
      Napi::Error::New(env, msg).ThrowAsJavaScriptException();
      return env.Null();
    }
    int d = dim_arg.As<Napi::Number>().Int32Value();
    if (env.IsExceptionPending())
      return env.Null();
    dynamic_dims.push_back(d < 0);
    if (d < 0)
      d = -d;
    dims.push_back(d);
  }

  auto shape = new xla::Shape(std::move(xla::ShapeUtil::MakeShape(
      primitive_type, absl::Span<const int64_t>(dims.data(), dims.size()),
      dynamic_dims)));
  return ShapeWrapper::Create(env, shape);
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
  auto [op] = match<xla::XlaOp *>(info, "XlaBuilder.build");
  if (env.IsExceptionPending())
    return env.Null();

  ASSIGN_OR_THROW(computation_instance, this->value()->Build(*op));

  return XlaComputationWrapper::Create(
      env, new xla::XlaComputation(std::move(computation_instance)));
}

namespace {

Napi::Value Parameter(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  auto [builder, parameter_number, shape, name] =
      match<xla::XlaBuilder *, int32_t, xla::Shape *, std::string>(info,
                                                                   "parameter");
  if (env.IsExceptionPending())
    return env.Null();
  xla::XlaOp *op =
      new xla::XlaOp(xla::Parameter(builder, parameter_number, *shape, name));
  return XlaOpWrapper::Create(env, op);
}

Napi::Value ConstantR0(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  const char *msg =
      "ConstantR0 expects a builder, a primtiive type and a number";
  if (info.Length() != 3 || !info[0].IsObject() || !info[1].IsString() ||
      !info[2].IsNumber()) {
    Napi::Error::New(env, msg).ThrowAsJavaScriptException();
    return env.Null();
  }

  auto builder = XlaBuilderWrapper::FromValue(env, info[0]);
  if (!builder)
    return env.Null();

  std::string ptype = info[1].As<Napi::String>().Utf8Value();

  xla::XlaOp *op = nullptr;
  if (ptype == "F32") {
    op = new xla::XlaOp(
        xla::ConstantR0<float>(builder, info[2].As<Napi::Number>()));
  }

  if (op == nullptr) {
    Napi::Error::New(env, msg).ThrowAsJavaScriptException();
    return env.Null();
  }

  return XlaOpWrapper::Create(env, op);
}

Napi::Value ConstantR1(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  const char *msg =
      "ConstantR0 expects a builder, a primtiive type and a number";
  if (info.Length() != 3 || !info[0].IsObject() || !info[1].IsString() ||
      !info[2].IsArray()) {
    Napi::Error::New(env, msg).ThrowAsJavaScriptException();
    return env.Null();
  }

  auto builder = XlaBuilderWrapper::FromValue(env, info[0]);
  if (!builder)
    return env.Null();

  std::string ptype = info[1].As<Napi::String>().Utf8Value();
  Napi::Array array = info[2].As<Napi::Array>();

  xla::XlaOp *op = nullptr;
  if (ptype == "F32") {
    std::vector<float> nums = floatArrayFromArray(env, array);
    if (env.IsExceptionPending())
      return env.Null();
    op = new xla::XlaOp(xla::ConstantR1<float>(
        builder, absl::Span<float>(nums.data(), nums.size())));
  }

  if (op == nullptr) {
    Napi::Error::New(env, msg).ThrowAsJavaScriptException();
    return env.Null();
  }

  return XlaOpWrapper::Create(env, op);
}

Napi::Value ConstantLiteral(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  auto [builder, literal] =
      match<xla::XlaBuilder *, xla::Literal *>(info, "constantLiteral");
  if (env.IsExceptionPending())
    return env.Null();
  auto op = new xla::XlaOp(xla::ConstantLiteral(builder, *literal));
  return XlaOpWrapper::Create(env, op);
}

Napi::Value Add(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  auto [lhs, rhs] = match<xla::XlaOp *, xla::XlaOp *>(info, "add");
  if (env.IsExceptionPending())
    return env.Null();
  xla::XlaOp *op = new xla::XlaOp(xla::Add(*lhs, *rhs));
  return XlaOpWrapper::Create(env, op);
}

Napi::Value Sub(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  auto [lhs, rhs] = match<xla::XlaOp *, xla::XlaOp *>(info, "sub");
  if (env.IsExceptionPending())
    return env.Null();
  xla::XlaOp *op = new xla::XlaOp(xla::Sub(*lhs, *rhs));
  return XlaOpWrapper::Create(env, op);
}

Napi::Value Max(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  auto [lhs, rhs] = match<xla::XlaOp *, xla::XlaOp *>(info, "max");
  if (env.IsExceptionPending())
    return env.Null();
  xla::XlaOp *op = new xla::XlaOp(xla::Max(*lhs, *rhs));
  return XlaOpWrapper::Create(env, op);
}

Napi::Value Mul(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  auto [lhs, rhs] = match<xla::XlaOp *, xla::XlaOp *>(info, "mul");
  if (env.IsExceptionPending())
    return env.Null();
  xla::XlaOp *op = new xla::XlaOp(xla::Mul(*lhs, *rhs));
  return XlaOpWrapper::Create(env, op);
}

Napi::Value Div(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  auto [lhs, rhs] = match<xla::XlaOp *, xla::XlaOp *>(info, "div");
  if (env.IsExceptionPending())
    return env.Null();
  xla::XlaOp *op = new xla::XlaOp(xla::Div(*lhs, *rhs));
  return XlaOpWrapper::Create(env, op);
}

Napi::Value DotGeneral(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  auto [lhs, rhs, lhs_contracting_dims, rhs_contracting_dims, lhs_batch_dims,
        rhs_batch_dims] =
      match<xla::XlaOp *, xla::XlaOp *, std::vector<int64_t>,
            std::vector<int64_t>, std::vector<int64_t>, std::vector<int64_t>>(
          info, "dotGeneral");
  if (env.IsExceptionPending())
    return env.Null();

  xla::DotDimensionNumbers dnums;
  for (int64_t i : lhs_contracting_dims)
    dnums.add_lhs_contracting_dimensions(i);
  for (int64_t i : rhs_contracting_dims)
    dnums.add_rhs_contracting_dimensions(i);
  for (int64_t i : lhs_batch_dims)
    dnums.add_lhs_batch_dimensions(i);
  for (int64_t i : rhs_batch_dims)
    dnums.add_rhs_batch_dimensions(i);

  xla::XlaOp *op = new xla::XlaOp(xla::DotGeneral(*lhs, *rhs, dnums));
  return XlaOpWrapper::Create(env, op);
}

Napi::Value Broadcast(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  auto [input, dims] =
      match<xla::XlaOp *, std::vector<int64_t>>(info, "broadcast");
  if (env.IsExceptionPending())
    return env.Null();
  xla::XlaOp *op = new xla::XlaOp(xla::Broadcast(
      *input, absl::Span<const int64_t>(dims.data(), dims.size())));
  return XlaOpWrapper::Create(env, op);
}

Napi::Value Transpose(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  auto [input, dims] =
      match<xla::XlaOp *, std::vector<int64_t>>(info, "transpose");
  if (env.IsExceptionPending())
    return env.Null();
  xla::XlaOp *op = new xla::XlaOp(xla::Transpose(
      *input, absl::Span<const int64_t>(dims.data(), dims.size())));
  return XlaOpWrapper::Create(env, op);
}

Napi::Value Reshape(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  auto [input, dims] =
      match<xla::XlaOp *, std::vector<int64_t>>(info, "reshape");
  if (env.IsExceptionPending())
    return env.Null();
  xla::XlaOp *op = new xla::XlaOp(xla::Reshape(
      *input, absl::Span<const int64_t>(dims.data(), dims.size())));
  return XlaOpWrapper::Create(env, op);
}

Napi::Value Iota(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  auto [builder, shape, index] =
      match<xla::XlaBuilder *, xla::Shape *, int64_t>(info, "iota");
  if (env.IsExceptionPending())
    return env.Null();
  xla::XlaOp *op = new xla::XlaOp(xla::Iota(builder, *shape, index));
  return XlaOpWrapper::Create(env, op);
}

Napi::Value Reduce(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  auto [builder, operands, init_values, computation, dimensions_to_reduce] =
      match<xla::XlaBuilder *, std::vector<xla::XlaOp>, std::vector<xla::XlaOp>,
            xla::XlaComputation *, std::vector<int64_t>>(info, "reduce");
  if (env.IsExceptionPending())
    return env.Null();

  if (operands.size() != init_values.size()) {
    Napi::Error::New(env, "reduce: thenumber of operands must be the same as "
                          "the number of init_values")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  xla::XlaOp *op = new xla::XlaOp(xla::Reduce(
      builder, absl::Span<const xla::XlaOp>(operands.data(), operands.size()),
      absl::Span<const xla::XlaOp>(init_values.data(), init_values.size()),
      *computation,
      absl::Span<const int64_t>(dimensions_to_reduce.data(),
                                dimensions_to_reduce.size())));
  return XlaOpWrapper::Create(env, op);
}

} // namespace

Napi::Object Init(Napi::Env env, Napi::Object exports) {
  Napi::HandleScope scope(env);

  PjRtClient::Init(env, exports);
  XlaBuilderWrapper::Init(env, exports);
  XlaOpWrapper::Init(env, exports);
  XlaComputationWrapper::Init(env, exports);
  PjRtLoadedExecutableWrapper::Init(env, exports);
  PjRtBufferWrapper::Init(env, exports);
  LiteralWrapper::Init(env, exports);
  ShapeWrapper::Init(env, exports);

  exports.Set(Napi::String::New(env, "constantR0"),
              Napi::Function::New<ConstantR0>(env));
  exports.Set(Napi::String::New(env, "constantR1"),
              Napi::Function::New<ConstantR1>(env));
  exports.Set(Napi::String::New(env, "constantLiteral"),
              Napi::Function::New<ConstantLiteral>(env));
  exports.Set(Napi::String::New(env, "parameter"),
              Napi::Function::New<Parameter>(env));
  exports.Set(Napi::String::New(env, "add"), Napi::Function::New<Add>(env));
  exports.Set(Napi::String::New(env, "max"), Napi::Function::New<Max>(env));
  exports.Set(Napi::String::New(env, "sub"), Napi::Function::New<Sub>(env));
  exports.Set(Napi::String::New(env, "mul"), Napi::Function::New<Mul>(env));
  exports.Set(Napi::String::New(env, "div"), Napi::Function::New<Div>(env));
  exports.Set(Napi::String::New(env, "dotGeneral"),
              Napi::Function::New<DotGeneral>(env));
  exports.Set(Napi::String::New(env, "broadcast"),
              Napi::Function::New<Broadcast>(env));
  exports.Set(Napi::String::New(env, "transpose"),
              Napi::Function::New<Transpose>(env));
  exports.Set(Napi::String::New(env, "reshape"),
              Napi::Function::New<Reshape>(env));
  exports.Set(Napi::String::New(env, "iota"), Napi::Function::New<Iota>(env));
  exports.Set(Napi::String::New(env, "reduce"),
              Napi::Function::New<Reduce>(env));

  return exports;
}

NODE_API_MODULE(testing, Init);