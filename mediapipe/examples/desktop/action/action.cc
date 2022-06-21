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
        input_stream: "in"
        output_stream: "out"

        node {
          calculator: "TfLiteConverterCalculator"
          input_stream: "MATRIX:in"
          output_stream: "TENSORS:image_tensor"
          options: {
            [mediapipe.TfLiteConverterCalculatorOptions.ext] {
              zero_center: false
            }
          }
        }


        node {
          calculator: "TfLiteInferenceCalculator"
          input_stream: "TENSORS:image_tensor"
          output_stream: "TENSORS:tensor_features"
          options: {
            [mediapipe.TfLiteInferenceCalculatorOptions.ext] {
              model_path: "mediapipe/models/adder_model_single_input_2x3.tflite"
            }
          }
        }

        node {
          calculator: "TfLiteTensorsToFloatsCalculator"
          input_stream: "TENSORS:tensor_features"
          output_stream: "FLOATS:out"
        }
        
      )pb");

  CalculatorGraph graph;
  LOG(INFO) << "MP_RETURN_IF_ERROR(graph.Initialize(config));";
  MP_RETURN_IF_ERROR(graph.Initialize(config));
  ASSIGN_OR_RETURN(OutputStreamPoller poller,
                   graph.AddOutputStreamPoller("out"));
  MP_RETURN_IF_ERROR(graph.StartRun({}));


  // Give 10 input packets that contain the vector [1.0, 1.0].
  LOG(INFO) <<"Matrix inputMatrix;";
  
  for (int i = 0; i < 10; ++i) {

    int nrows = 2;
    int ncols = 3;
    Matrix inputMatrix;
    inputMatrix.resize(nrows, ncols);
    for (size_t i = 0; i < nrows; i++)
    {
        for (size_t j = 0; j < ncols; j++)
        {
            int index = i * ncols + j;
            inputMatrix(i, j) = (float) index;
            LOG(INFO) << "index: " << index << " inputMatrix(i, j): " << inputMatrix(i, j);
        }
    }

    MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
        "in", MakePacket<Matrix>(inputMatrix).At(Timestamp(i))));
  }
  // Close the input stream "in".
  MP_RETURN_IF_ERROR(graph.CloseInputStream("in"));
  mediapipe::Packet packet;

  // Get output packets.
  LOG(INFO) << "Get output packets";
  int counter = 0;
  while (poller.Next(&packet)) {
    auto outputMatrix = packet.Get<std::vector<float>>();
    counter ++;
    LOG(INFO) << "Counter: " << counter ;
    // LOG(INFO) << "outputMatrix: " << outputMatrix ;
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
