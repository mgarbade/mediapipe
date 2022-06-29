licenses(["notice"])

package(default_visibility = ["//mediapipe/examples:__subpackages__"])

cc_binary(
    name = "action",
    srcs = ["action.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/calculators/tflite:tflite_converter_calculator",
        "//mediapipe/calculators/tflite:tflite_inference_calculator",
        "//mediapipe/calculators/tflite:tflite_tensors_to_floats_calculator",
        
        # convert OpenCV mat to Eigen Mat
        "//mediapipe/framework/port:opencv_core",

        # "//mediapipe/calculators/tensorflow:matrix_to_tensor_calculator",
        # "//mediapipe/calculators/tensorflow:tensor_to_matrix_calculator",
        # "//mediapipe/calculators/core:matrix_to_vector_calculator",

        "//mediapipe/calculators/core:pass_through_calculator",

        "//mediapipe/framework:calculator_graph",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status",
    ],
)
