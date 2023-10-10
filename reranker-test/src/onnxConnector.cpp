
#define NOMINMAX
#include <chrono>
#include "MemAlignedTensor.h"
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

    std::vector<std::vector<float>> CompareSentences(const std::vector<std::vector<int>>& sentences, int eosId) override try {
        const auto startTime = std::chrono::system_clock::now();
        EnsureInitialized();

        // getting max token size
        auto maxTokenSize = sentences[0].size();
        for (const auto& sentence : sentences) {
            maxTokenSize = std::max(maxTokenSize, sentence.size());
        }

        // allocate token and attention-mask matrix
        std::vector<int64_t> tokenArray(maxTokenSize * sentences.size(), 0LL);
        std::vector<int64_t> attentionMaskArray(maxTokenSize * sentences.size(), 0LL);

        // setup token and attention-mask matrix
        for (size_t i = 0; i < sentences.size(); ++i) {
            const auto& sentence = sentences[i];
            auto tokenTop = &tokenArray[i * maxTokenSize];
            auto maskTop = &attentionMaskArray[i * maxTokenSize];
            for (size_t j = 0; j < sentence.size(); ++j) {
                tokenTop[j] = sentence[j];
                maskTop[j] = 1LL;
            }
        }

        // finding different token index
        size_t compareStartPoint = 0;
        for (size_t i = 0; i < maxTokenSize && compareStartPoint == 0; ++i) {
            if (i >= sentences[0].size()) {
                compareStartPoint = i - 1;
                break;
            }
            const auto targetToken = sentences[0][i];
            for (size_t j = 1; j < sentences.size(); ++j) {
                if (i >= sentences[j].size() || targetToken != sentences[j][i]) {
                    compareStartPoint = i - 1;
                    break;
                }
            }
        }

        // binding input
        m_binding.Clear();
        const auto& inputIdsTensor = winrt::TensorInt64Bit::CreateFromArray({ sentences.size(), maxTokenSize }, tokenArray);
        m_binding.Bind(L"input_ids", inputIdsTensor);

        const auto& attentionTensor = winrt::TensorInt64Bit::CreateFromArray({ sentences.size(), maxTokenSize }, attentionMaskArray);
        m_binding.Bind(L"attention_mask", attentionTensor);

        const auto& results = m_session.Evaluate(m_binding, L"correlationId");
        const auto evaluateEndTime = std::chrono::system_clock::now();

        const auto& resultOutput = results.Outputs().Lookup(L"logits").as<winrt::TensorFloat>();
        const auto& outputShape = resultOutput.Shape();
        if (outputShape.Size() != 3) throw std::runtime_error("unexpected shape");

        const auto batchCount = outputShape.GetAt(0);
        const auto sequenceCount = outputShape.GetAt(1);
        const auto tokenCount = outputShape.GetAt(2);

        std::vector<std::vector<float>> result;

        MemAlignedTensor tokenVector;
        tokenVector.Reserve(1, tokenCount);
        auto [bufferPtr, _rowSize, _columnSize] = tokenVector.GetBuffer();
        auto tokenVectorView = winrt::array_view<float>(bufferPtr, bufferPtr + tokenCount);

        for (size_t i = 0; i < sentences.size(); ++i) {
            std::vector<float> probList;
            probList.emplace_back(1.0f);
            for (size_t tokenIndex = 1; tokenIndex < sentences[i].size(); ++tokenIndex) {
                const auto targetVectorOffset = i * sequenceCount * tokenCount + (tokenIndex - 1) * tokenCount;
                resultOutput.GetAsVectorView().GetMany(targetVectorOffset, tokenVectorView);
                const auto probability = tokenVector.GetProbability(sentences[i][tokenIndex]);
                probList.emplace_back(probability);
            }
            const auto targetVectorOffset = i * sequenceCount * tokenCount + (sentences[i].size() - 1) * tokenCount;
            resultOutput.GetAsVectorView().GetMany(targetVectorOffset, tokenVectorView);
            const auto probability = tokenVector.GetProbability(eosId);
            probList.emplace_back(probability);

            result.emplace_back(probList);
        }

        return result;
    }
    catch (...) { return std::vector<std::vector<float>>(); }

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

