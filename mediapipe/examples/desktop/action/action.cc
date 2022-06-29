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
#include <opencv2/core/eigen.hpp>

#include <iostream>
#include <vector>
#include <iterator>
#include <fstream>
#include <sstream>
#include <string>
#include <iomanip> // set precision of output string
#include <opencv2/core/core.hpp> // OpenCV matrices for storing data

using namespace std;
using namespace cv;

namespace mediapipe {

void readMatAsciiWithHeader( const string& filename, Mat& matData)
{
    cout << "Create matrix from file :" << filename << endl;

    ifstream inFileStream(filename.c_str());
    if(!inFileStream){
        cout << "File cannot be found" << endl;
        exit(-1);
    }

    int rows, cols;
    inFileStream >> rows;
    inFileStream >> cols;
    matData.create(rows,cols,CV_32F);
    cout << "numRows: " << rows << "\t numCols: " << cols << endl;

    matData.setTo(0);  // init all values to 0
    float *dptr;
    for(int ridx=0; ridx < matData.rows; ++ridx){
        dptr = matData.ptr<float>(ridx);
        for(int cidx=0; cidx < matData.cols; ++cidx, ++dptr){
            inFileStream >> *dptr;
        }
    }
    inFileStream.close();

}

absl::Status PrintNetworkOutput() {

  LOG(INFO) << "Read data from ascii";
  // string filename = "/media/data_ssd/libs/mediapipe_v0.8.9/notebooks/squat_neg_10x36.mat";
  string filename = "/media/data_ssd/libs/mediapipe_v0.8.9/notebooks/squat_trans_36x10.mat";
  // string filename = "/media/data_ssd/libs/mediapipe_v0.8.9/notebooks/skeletons_with_neck_squat_trans_36x79.mat";
  Mat matData;
  readMatAsciiWithHeader( filename, matData);

  // convert opencv mat to eigen mat
  // Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> matrixEigen(matData.ptr<float>(), matData.rows, matData.cols);


  LOG(INFO) << "PrintNetworkOutput";
  // Configures a simple graph, which concatenates 2 PassThroughCalculators.
  CalculatorGraphConfig config =
      ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_stream: "MATRIX:in"
        output_stream: "FLOATS:out"

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
              model_path: "notebooks/model_ar_simple_squat_only.tflite"
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

  std::vector<Matrix> inputMatrices;

  for (int i = 0; i < 100; ++i) {

    int nrows = matData.rows;
    int ncols = matData.cols;
    Matrix inputMatrix;
    inputMatrix.resize(nrows, ncols);
    for (size_t i = 0; i < nrows; i++)
    {
        for (size_t j = 0; j < ncols; j++)
        {
            int index = i * ncols + j;
            inputMatrix(i, j) = matData.at<float>(i, j);
            // LOG(INFO) << "index: " << index << " inputMatrix(i, j): " << inputMatrix(i, j);
        }
    }
    inputMatrices.push_back(inputMatrix);

  }

  for (size_t i = 0; i < 10; i++)
  {
    MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
      "in", MakePacket<Matrix>(inputMatrices.at(i)).At(Timestamp(i))));
  }



  // Close the input stream "in".
  MP_RETURN_IF_ERROR(graph.CloseInputStream("in"));
  mediapipe::Packet packet;

  // Get output packets.
  LOG(INFO) << "Get output packets";
  int counter = 0;
  while (counter < 100) {

    if (poller.Next(&packet)){
      auto outputMatrix = packet.Get<std::vector<float>>();
      LOG(INFO) << "outputMatrix: ";
      for (auto item : outputMatrix)
        LOG(INFO) << item << ", " ;
    }
    else{
      LOG(INFO) << "Poller could not get package" ;
    }

    counter ++;
    LOG(INFO) << "Counter: " << counter;
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
