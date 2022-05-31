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

#include <iostream>
#include <vector>

namespace mediapipe {

absl::Status PrintNetworkOutput() {

  std::cout << "PrintNetworkOutput" << std::endl;
  // Configures a simple graph, which concatenates 2 PassThroughCalculators.
  CalculatorGraphConfig config =
      ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_stream: "in"
        output_stream: "out"
        node {
          calculator: "ActionCalculator"
          input_stream: "INPUT:in"
          output_stream: "OUTPUT:out"
        }
      )pb");

  CalculatorGraph graph;
  std::cout << "MP_RETURN_IF_ERROR(graph.Initialize(config));" << std::endl;
  MP_RETURN_IF_ERROR(graph.Initialize(config));
  ASSIGN_OR_RETURN(OutputStreamPoller poller,
                   graph.AddOutputStreamPoller("out"));
  MP_RETURN_IF_ERROR(graph.StartRun({}));


  // Give 10 input packets that contain the vector [1.0, 1.0].
  std::cout << "std::vector<float> inputVector;" << std::endl;
  std::vector<float> inputVector;
  inputVector.push_back(1.0);
  inputVector.push_back(2.0);

  std::string outputString1 = std::to_string(inputVector[0]);
  std::string outputString2 = std::to_string(inputVector[1]);
  
  std::cout << "outputString1: " << outputString1 << std::endl;
  std::cout << "outputString2: " << outputString2 << std::endl;

  for (int i = 0; i < 10; ++i) {
    MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
        "in", MakePacket<std::vector<float>>(inputVector).At(Timestamp(i))));
  }
  // Close the input stream "in".
  MP_RETURN_IF_ERROR(graph.CloseInputStream("in"));
  mediapipe::Packet packet;

  // Get the output packets string.
  while (poller.Next(&packet)) {
    auto outputVectorFloat = packet.Get<std::vector<float>>();
    std::string outputString1 = std::to_string(outputVectorFloat[0]);
    std::string outputString2 = std::to_string(outputVectorFloat[1]);
    LOG(INFO) << outputString1;
    LOG(INFO) << outputString2;
  }
  return graph.WaitUntilDone();
}
}  // namespace mediapipe

int main(int argc, char** argv) {
  std::cout << "Start main function" << std::endl;
  google::InitGoogleLogging(argv[0]);
  std::cout << "Start Mediapipe" << std::endl;
  CHECK(mediapipe::PrintNetworkOutput().ok());
  return 0;
}
