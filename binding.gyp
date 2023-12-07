{
  "targets": [
    {
      "target_name": "xla",
      'include_dirs': [
        "<!@(node -p \"require('node-addon-api').include\")",
        ".",
        "/usr/local/include",
        "/home/jarin/projects/xla-rs/xla_extension/include"
       ],
       'dependencies': ["<!(node -p \"require('node-addon-api').gyp\")"],
      "cflags" : [
        "-std=c++17",
        "-Wno-deprecated-declarations",
      ],
      'cflags!': [ '-fno-exceptions', "-fno-rtti", '-D_THREAD_SAFE' ],
      "cflags_cc!": [ "-fno-rtti", "-fno-exceptions" ],
       "libraries": [
           "<!(node ./libs)"
      ],
      "sources": [
        "xla_js.cc",
        "xla_js.h"
      ],
    }
  ]
}