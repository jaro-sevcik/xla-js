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

#endif // XLA_JS_H_
