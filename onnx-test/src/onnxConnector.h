#pragma once
#include <memory>
#include <string_view>
#include <vector>

struct OnnxConnector {
    virtual void Initialize(const std::wstring_view modelFile) = 0;
    virtual std::tuple<int64_t, float> GetPrediction(const std::vector<int64_t>& tokens) = 0;

    virtual ~OnnxConnector() {};
    static std::shared_ptr<OnnxConnector> CreateInstance();
};
