#include "register_onnxruntime.h"

#include "rotated_nms.h"
#include "ort_utils.h"
const char *c_MyOpDomain = "yolox";
RotatedNmsOp c_RotatedNmsOp;

OrtStatus *ORT_API_CALL RegisterCustomOps(OrtSessionOptions *options,
                                          const OrtApiBase *api) {
  OrtCustomOpDomain *domain = nullptr;
  const OrtApi *ortApi = api->GetApi(ORT_API_VERSION);

  if (auto status = ortApi->CreateCustomOpDomain(c_MyOpDomain, &domain)) {
    return status;
  }

  if (auto status = ortApi->CustomOpDomain_Add(domain, &c_RotatedNmsOp)) {
    return status;
  }
  return ortApi->AddCustomOpDomain(options, domain);
}