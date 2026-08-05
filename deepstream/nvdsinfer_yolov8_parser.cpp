/*
 * Ultralytics YOLOv8 raw ONNX parser for DeepStream nvinfer.
 * Expects output tensor shaped [1, 4+num_classes, num_anchors] (e.g. 1x84x8400).
 */

#include "nvdsinfer_custom_impl.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

extern "C" bool NvDsInferParseYoloV8(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList);

static float clampf(float v, float lo, float hi)
{
  return std::max(lo, std::min(v, hi));
}

static void decodeYoloV8(
    const float* data,
    int channels,
    int numAnchors,
    int netW,
    int netH,
    int numClasses,
    const std::vector<float>& preclusterThreshold,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
  // Layout: [cx, cy, w, h, class0, class1, ...] over channels, anchors last.
  const int boxChannels = 4;
  if (channels < boxChannels + 1) {
    std::cerr << "YoloV8 parser: unexpected channel count " << channels << std::endl;
    return;
  }
  if (numClasses <= 0) {
    numClasses = channels - boxChannels;
  }
  numClasses = std::min(numClasses, channels - boxChannels);

  objectList.clear();
  objectList.reserve(256);

  for (int a = 0; a < numAnchors; ++a) {
    const float cx = data[0 * numAnchors + a];
    const float cy = data[1 * numAnchors + a];
    const float bw = data[2 * numAnchors + a];
    const float bh = data[3 * numAnchors + a];

    int bestCls = 0;
    float bestScore = data[(boxChannels + 0) * numAnchors + a];
    for (int c = 1; c < numClasses; ++c) {
      const float s = data[(boxChannels + c) * numAnchors + a];
      if (s > bestScore) {
        bestScore = s;
        bestCls = c;
      }
    }

    const float thr =
        (bestCls < static_cast<int>(preclusterThreshold.size()))
            ? preclusterThreshold[bestCls]
            : 0.25f;
    if (bestScore < thr) {
      continue;
    }

    float x1 = cx - 0.5f * bw;
    float y1 = cy - 0.5f * bh;
    float x2 = cx + 0.5f * bw;
    float y2 = cy + 0.5f * bh;

    x1 = clampf(x1, 0.f, static_cast<float>(netW));
    y1 = clampf(y1, 0.f, static_cast<float>(netH));
    x2 = clampf(x2, 0.f, static_cast<float>(netW));
    y2 = clampf(y2, 0.f, static_cast<float>(netH));

    const float w = x2 - x1;
    const float h = y2 - y1;
    if (w < 1.f || h < 1.f) {
      continue;
    }

    NvDsInferParseObjectInfo obj{};
    obj.left = x1;
    obj.top = y1;
    obj.width = w;
    obj.height = h;
    obj.detectionConfidence = bestScore;
    obj.classId = bestCls;
    objectList.push_back(obj);
  }
}

extern "C" bool NvDsInferParseYoloV8(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
  if (outputLayersInfo.empty()) {
    std::cerr << "YoloV8 parser: no output layers" << std::endl;
    return false;
  }

  const NvDsInferLayerInfo& out = outputLayersInfo[0];
  const int ndims = out.inferDims.numDims;
  if (ndims < 2) {
    std::cerr << "YoloV8 parser: bad dims count " << ndims << std::endl;
    return false;
  }

  // Accept [C, A], [1, C, A], or [B, C, A] with B==1.
  int channels = 0;
  int numAnchors = 0;
  if (ndims == 2) {
    channels = out.inferDims.d[0];
    numAnchors = out.inferDims.d[1];
  } else if (ndims == 3) {
    // Prefer channels < anchors (84 < 8400).
    if (out.inferDims.d[1] < out.inferDims.d[2]) {
      channels = out.inferDims.d[1];
      numAnchors = out.inferDims.d[2];
    } else {
      // Unexpected transposed [1, A, C]
      channels = out.inferDims.d[2];
      numAnchors = out.inferDims.d[1];
      std::cerr << "YoloV8 parser: unexpected [B,A,C] layout; unsupported" << std::endl;
      return false;
    }
  } else {
    std::cerr << "YoloV8 parser: unsupported numDims=" << ndims << std::endl;
    return false;
  }

  const int numClasses = detectionParams.numClassesConfigured > 0
                             ? static_cast<int>(detectionParams.numClassesConfigured)
                             : (channels - 4);

  decodeYoloV8(
      static_cast<const float*>(out.buffer),
      channels,
      numAnchors,
      static_cast<int>(networkInfo.width),
      static_cast<int>(networkInfo.height),
      numClasses,
      detectionParams.perClassPreclusterThreshold,
      objectList);

  return true;
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseYoloV8);
