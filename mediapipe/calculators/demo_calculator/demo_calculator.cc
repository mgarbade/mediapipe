#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/status.h"


// #include <memory>
#include <string>
// #include <vector>

// #include "absl/strings/str_split.h"
// #include "absl/strings/string_view.h"
// #include "absl/strings/strip.h"

#include "mediapipe/calculators/demo_calculator/demo_calculator.pb.h"


namespace mediapipe {

class ExampleCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc);

  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;
  absl::Status Close(CalculatorContext* cc) override;

};

absl::Status ExampleCalculator::GetContract(CalculatorContract* cc) {
  cc->Inputs().Tag("INPUT").Set<std::string>();
  cc->Outputs().Tag("OUTPUT").Set<std::string>();
  return absl::OkStatus();
}

absl::Status ExampleCalculator::Open(CalculatorContext* cc) {
  const auto& options = cc->Options<::mediapipe::ExampleCalculatorOptions>();
  auto var = options.option_parameter_1();
  return absl::OkStatus();
}

absl::Status ExampleCalculator::Process(CalculatorContext* cc) {
  if (!cc->Inputs().Tag("INPUT").IsEmpty())
  {
    auto stringInput = cc->Inputs().Tag("INPUT").Get<std::string>();
    auto stringOutput = stringInput + " - my additional text element";

    Packet data = MakePacket<std::string>(stringOutput);
    Timestamp outputTimestamp = cc->InputTimestamp();
    Packet dataWithTimestamp = data.At(outputTimestamp);
    cc->Outputs().Tag("OUTPUT").AddPacket(dataWithTimestamp);
    //cc->Outputs().Tag("OUTPUT").AddPacket(cc->Inputs().Tag("INPUT").Value());
  }
  return absl::OkStatus();
}

absl::Status ExampleCalculator::Close(CalculatorContext* cc) {
  return absl::OkStatus();
}

REGISTER_CALCULATOR(ExampleCalculator);

}  // namespace mediapipe
