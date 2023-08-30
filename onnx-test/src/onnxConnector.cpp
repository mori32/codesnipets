
#include <chrono>
#include "onnxConnector.h"
#include <winrt/Windows.AI.MachineLearning.h>
#include <winrt/Windows.Foundation.Collections.h>
#include <winrt/Windows.Storage.h>
#include "miscUtils.h"

namespace winrt {
    using namespace ::winrt::Windows::AI::MachineLearning;
    using namespace ::winrt::Windows::Foundation::Collections;
    using namespace ::winrt::Windows::Storage;
}

struct OnnxConnectorImpl : public OnnxConnector {
public:
    void Initialize(const std::wstring_view modelFileName) {
        m_modelFileName = modelFileName;
    }

    std::tuple<int64_t, float> GetPrediction(const std::vector<int64_t>& tokens) override try {
        const auto startTime = std::chrono::system_clock::now();
        EnsureInitialized();

        // create attention-mask, that is filled by '1' where token vector has token value.
        const auto tokenSize = static_cast<int64_t>(tokens.size());
        std::vector<int64_t> attentionMask(tokens.size(), 1LL);

        // binding input
        m_binding.Clear();
        const auto& inputIdsTensor = winrt::TensorInt64Bit::CreateFromArray({ 1, tokenSize }, tokens);
        m_binding.Bind(L"input_ids", inputIdsTensor);

        const auto& attentionTensor = winrt::TensorInt64Bit::CreateFromArray({ 1, tokenSize }, attentionMask);
        m_binding.Bind(L"attention_mask", attentionTensor);

        const auto& results = m_session.Evaluate(m_binding, L"correlationId");
        const auto evaluateEndTime = std::chrono::system_clock::now();

        const auto& resultOutput = results.Outputs().Lookup(L"logits").as<winrt::TensorFloat>();
        const auto& outputShape = resultOutput.Shape();
        if (outputShape.Size() != 3) throw std::runtime_error("unexpected shape");

        const auto batchCount = outputShape.GetAt(0);
        const auto sequenceCount = outputShape.GetAt(1);
        const auto tokenCount = outputShape.GetAt(2);

        const auto targetVectorOffset = (sequenceCount - 1) * tokenCount;
        std::vector<float> logitsLast(tokenCount, 0.0f);

        resultOutput.GetAsVectorView().GetMany(static_cast<uint32_t>(targetVectorOffset),
            winrt::array_view<float>(logitsLast));

        const auto maxIndex = FindMaxIndex(logitsLast);

        return std::make_tuple(maxIndex, logitsLast[maxIndex]); // TODO: should softmax
    }
    catch (...) { return std::make_tuple(-1LL, 0.0f); }


private:
    void EnsureInitialized() {
        if (!m_model) {
            const auto& utf8FileName = ToUtf8(m_modelFileName);

            m_model = winrt::LearningModel::LoadFromFilePath(m_modelFileName);
            const auto deviceKind = winrt::LearningModelDeviceKind::Cpu;
            m_session = winrt::LearningModelSession{ m_model, winrt::LearningModelDevice(deviceKind) };
            m_binding = winrt::LearningModelBinding{ m_session };
        }
    }

    int64_t FindMaxIndex(const std::vector<float>& list) {
        size_t resultIndex = 0;
        float maxLogit = list[0];
        for (size_t i = 1; i < list.size(); ++i) {
            if (maxLogit < list[i]) {
                maxLogit = list[i];
                resultIndex = i;
            }
        }
        return static_cast<int64_t>(resultIndex);
    }

private:
    std::wstring m_modelFileName;
    winrt::LearningModel m_model{ nullptr };
    winrt::LearningModelSession m_session{ nullptr };
    winrt::LearningModelBinding m_binding{ nullptr };
};

std::shared_ptr<OnnxConnector> OnnxConnector::CreateInstance() {
    return std::make_shared<OnnxConnectorImpl>();
}

