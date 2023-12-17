{
  "targets": [
    {
      "target_name": "xla",
      'include_dirs': [
        "<!@(node -p \"require('node-addon-api').include\")",
        "./cpp",
        "./xla_extension/include",
        "/usr/local/include"
       ],
       'dependencies': ["<!(node -p \"require('node-addon-api').gyp\")"],
      "cflags" : [
        "-std=c++17",
        "-Wno-deprecated-declarations",
      ],
      'cflags!': [ '-fno-exceptions', "-fno-rtti", '-D_THREAD_SAFE' ],
      "cflags_cc!": [ "-fno-rtti", "-fno-exceptions" ],
       "libraries": [
           "<!(node ./scripts/libs)"
      ],
      "sources": [
        "cpp/xla_js.cc",
        "cpp/xla_js.h"
      ],
    }
  ]
}