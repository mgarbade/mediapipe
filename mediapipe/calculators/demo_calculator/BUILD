licenses(["notice"])

package(default_visibility = ["//visibility:private"])

load("//mediapipe/framework/port:build_config.bzl", "mediapipe_cc_proto_library")

proto_library(
    name = "demo_calculator_proto",
    srcs = ["demo_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_proto",
    ],
)

mediapipe_cc_proto_library(
    name = "demo_calculator_cc_proto",
    srcs = ["demo_calculator.proto"],
    cc_deps = ["//mediapipe/framework:calculator_cc_proto"],
    visibility = ["//visibility:public"],
    deps = [":demo_calculator_proto"],
)


cc_library(
    name = "demo_calculator",
    srcs = ["demo_calculator.cc"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/port:status",
        ":demo_calculator_cc_proto"
    ],
    alwayslink = 1,
)



