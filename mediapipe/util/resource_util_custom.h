#ifndef MEDIAPIPE_UTIL_RESOURCE_UTIL_CUSTOM_H_
#define MEDIAPIPE_UTIL_RESOURCE_UTIL_CUSTOM_H_

#include <string>

#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/statusor.h"

namespace mediapipe {

typedef std::function<absl::Status(const std::string&, std::string*)>
    ResourceProviderFn;

typedef std::function<absl::StatusOr<std::string>(const std::string&)>
    PathResolverFn;

// Overrides the behavior of GetResourceContents.
void SetCustomGlobalResourceProvider(ResourceProviderFn fn);

// Overfides the behavior of PathToResourceAsFile.
void SetCustomGlobalPathResolver(PathResolverFn fn);

}  // namespace mediapipe

#endif  // MEDIAPIPE_UTIL_RESOURCE_UTIL_CUSTOM_H_
