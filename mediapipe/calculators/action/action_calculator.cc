#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/status.h"

//#include "tensorflow/lite/interpreter.h"


// #include <memory>
#include <string>
#include <vector>

// #include "absl/strings/str_split.h"
// #include "absl/strings/string_view.h"
// #include "absl/strings/strip.h"

// #include "mediapipe/calculators/action/action.pb.h"


namespace mediapipe {

// Tag names of input and output nodes
namespace{
  constexpr char InputVectorFloat[] = "INPUT";
  constexpr char OutputVectorFloat[] = "VECTOR_FLOAT";
}

class ActionCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc);

  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;
  absl::Status Close(CalculatorContext* cc) override;

};

absl::Status ActionCalculator::GetContract(CalculatorContract* cc) {
  LOG(INFO) << "ActionCalculator::GetContract(CalculatorContract* cc)";
  cc->Inputs().Tag(InputVectorFloat).Set<std::vector<float>>();
  cc->Outputs().Tag(OutputVectorFloat).Set<std::vector<float>>();
  return absl::OkStatus();
}

absl::Status ActionCalculator::Open(CalculatorContext* cc) {
  // const auto& options = cc->Options<::mediapipe::ActionCalculatorOptions>();
  // auto var = options.option_parameter_1();
  return absl::OkStatus();
}

absl::Status ActionCalculator::Process(CalculatorContext* cc) {
  LOG(INFO) << "ActionCalculator::Process";
  if (!cc->Inputs().Tag(InputVectorFloat).IsEmpty())
  {
    auto inputVectorFloat = cc->Inputs().Tag(InputVectorFloat).Get<std::vector<float>>();
    
    // make NN forward pass / inference here


    Packet data = MakePacket<std::vector<float>>(inputVectorFloat);
    Timestamp outputTimestamp = cc->InputTimestamp();
    Packet dataWithTimestamp = data.At(outputTimestamp);
    cc->Outputs().Tag(OutputVectorFloat).AddPacket(dataWithTimestamp);
  }
  return absl::OkStatus();
}

absl::Status ActionCalculator::Close(CalculatorContext* cc) {
  return absl::OkStatus();
}

REGISTER_CALCULATOR(ActionCalculator);

}  // namespace mediapipe
