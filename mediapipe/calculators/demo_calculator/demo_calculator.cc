#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/status.h"


namespace mediapipe {

class ExampleCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc);

  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;
  absl::Status Close(CalculatorContext* cc) override;

};

absl::Status ExampleCalculator::GetContract(CalculatorContract* cc) {
  cc->Inputs().Tag("TAG_NAME").Set<std::string>();
  cc->Outputs().Tag("OUTPUT_TAG_NAME").Set<std::string>();
  return absl::OkStatus();
}

absl::Status ExampleCalculator::Open(CalculatorContext* cc) {
  return absl::OkStatus();
}

absl::Status ExampleCalculator::Process(CalculatorContext* cc) {
  return absl::OkStatus();
}

absl::Status ExampleCalculator::Close(CalculatorContext* cc) {
  return absl::OkStatus();
}

REGISTER_CALCULATOR(ExampleCalculator);

}  // namespace mediapipe
