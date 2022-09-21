#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/statusor.h"

namespace mediapipe {

using mediapipe::file::GetContents;

namespace internal {

absl::Status DefaultGetResourceContents(const std::string& path,
                                        std::string* output,
                                        bool read_as_binary) {
  return GetContents(path, output, read_as_binary);
}

absl::StatusOr<std::string> DefaultPathToResourceAsFile(const std::string& path) {
  return path;
}

}  // namespace internal
}  // namespace mediapipe
