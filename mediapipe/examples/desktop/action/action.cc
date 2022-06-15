// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// A simple example to print out Vector of floats from a MediaPipe graph.

#include "mediapipe/framework/calculator_graph.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/formats/matrix.h"

#include <iostream>
#include <vector>

namespace mediapipe {

absl::Status PrintNetworkOutput() {

  LOG(INFO) << "PrintNetworkOutput";
  // Configures a simple graph, which concatenates 2 PassThroughCalculators.
  CalculatorGraphConfig config =
      ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_stream: "INPUT:in"
        output_stream: "TENSORS:tflite_prediction"

        node {
          calculator: "ActionCalculator"
          input_stream: "INPUT:in"
          output_stream: "VECTOR_FLOAT:vector_float"
        }

        node {
          calculator: "VectorToTensorCalculator"
          input_stream: "VECTOR_FLOAT:vector_float"
          output_stream: "MATRIX:matrix"
        }

        node {
          calculator: "TfLiteConverterCalculator"
          input_stream: "MATRIX:matrix"
          output_stream: "TENSORS:landmark_tensors"
        }

        node {
          calculator: "TfLiteInferenceCalculator"
          input_stream: "TENSORS:landmark_tensors"
          output_stream: "TENSORS:tflite_prediction"
          node_options: {
            [type.googleapis.com/mediapipe.TfLiteInferenceCalculatorOptions] {
              model_path: "mediapipe/models/adder_model_single_input_2x3.tflite"
              delegate { xnnpack {} }
            }
          }
        }

      )pb");

  CalculatorGraph graph;
  LOG(INFO) << "MP_RETURN_IF_ERROR(graph.Initialize(config));";
  MP_RETURN_IF_ERROR(graph.Initialize(config));
  ASSIGN_OR_RETURN(OutputStreamPoller poller,
                   graph.AddOutputStreamPoller("tflite_prediction"));
  MP_RETURN_IF_ERROR(graph.StartRun({}));


  // Give 10 input packets that contain the vector [1.0, 1.0].
  LOG(INFO) <<"std::vector<float> inputVector;";
  std::vector<float> inputVector;
  for (size_t i = 0; i < 6; i++)
  {
    inputVector.push_back((float) i + 1);
    // LOG(INFO) <<"inputVector[i]: " << std::to_string(inputVector[i]);
  }
  
  for (int i = 0; i < 10; ++i) {
    MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
        "in", MakePacket<std::vector<float>>(inputVector).At(Timestamp(i))));
  }
  // Close the input stream "in".
  MP_RETURN_IF_ERROR(graph.CloseInputStream("in"));
  mediapipe::Packet packet;

  // Get output packets.
  LOG(INFO) << "Get output packets";
  while (poller.Next(&packet)) {
    auto outputMatrix = packet.Get<Matrix>();
    // LOG(INFO) << "outputMatrix: " << outputMatrix 
    //           << " outputMatrix.size:" << outputMatrix.size()
    //           << " outputMatrix.rows:" << outputMatrix.rows()
    //           << " outputMatrix.cols:" << outputMatrix.cols();
    // std::vector<float> outputVectorFloat;

    // outputVectorFloat.push_back(outputMatrix(0, 0));
    // outputVectorFloat.push_back(outputMatrix(0, 1));

    // std::string outputString1 = std::to_string(outputVectorFloat[0]);
    // std::string outputString2 = std::to_string(outputVectorFloat[1]);
    // LOG(INFO) << outputString1;
    // LOG(INFO) << outputString2;
  }
  return graph.WaitUntilDone();
}
}  // namespace mediapipe

int main(int argc, char** argv) {
  LOG(INFO) << "Start main function";
  google::InitGoogleLogging(argv[0]);
  LOG(INFO) << "Start Mediapipe";
  CHECK(mediapipe::PrintNetworkOutput().ok());
  return 0;
}
