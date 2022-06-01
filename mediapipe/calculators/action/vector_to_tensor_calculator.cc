#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/formats/matrix.h"

#include <vector>


namespace mediapipe{

    namespace{
        constexpr char VectorFloat[] = "INPUT";
        constexpr char OutputMatrix[] = "OUTPUT";
    }

    class VectorToTensorCalculator : public CalculatorBase {
        public:
            static absl::Status GetContract(CalculatorContract* cc);

            absl::Status Open(CalculatorContext* cc) override;
            absl::Status Process(CalculatorContext* cc) override;
            absl::Status Close(CalculatorContext* cc) override;

        };

        absl::Status VectorToTensorCalculator::GetContract(CalculatorContract* cc){
            LOG(INFO) << "GetContract";
            cc->Inputs().Tag(VectorFloat).Set<std::vector<float>>();
            cc->Outputs().Tag(OutputMatrix).Set<Matrix>();
            LOG(INFO) << "GetContract Completed";
            return absl::OkStatus();
        }
        absl::Status VectorToTensorCalculator::Open(CalculatorContext* cc){
            LOG(INFO) << "Open";
            return absl::OkStatus();
        }
        absl::Status VectorToTensorCalculator::Process(CalculatorContext* cc){
            LOG(INFO) << "Process";
            std::vector<float> inputVectorFloat = cc->Inputs().Tag(VectorFloat).Get<std::vector<float>>();

            LOG(INFO) << "Matrix";
            Matrix matrix;

            LOG(INFO) << "matrix.resize(2, 1);";
            matrix.resize(2, 1);

            matrix(0, 0) = inputVectorFloat.at(0);
            matrix(1, 0) = inputVectorFloat.at(1);

            LOG(INFO) << "std::unique_ptr<Matrix> output_stream_collection = std::make_unique<Matrix>(matrix); ";
            std::unique_ptr<Matrix> output_stream_collection = std::make_unique<Matrix>(matrix); 
            LOG(INFO) << "cc -> Outputs().Tag(OutputMatrix).Add(output_stream_collection.release(), cc->InputTimestamp());";
            cc -> Outputs().Tag(OutputMatrix).Add(output_stream_collection.release(), cc->InputTimestamp());
            return absl::OkStatus();
        }
        absl::Status VectorToTensorCalculator::Close(CalculatorContext* cc){
            return absl::OkStatus();
        }

    REGISTER_CALCULATOR(VectorToTensorCalculator);
}