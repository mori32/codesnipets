
#define NOMINMAX
#include <chrono>
#include <sstream>
#include <onnxruntime_cxx_api.h>
#include "MemAlignedTensor.h"
#include "onnxConnector.h"
#include "miscUtils.h"

struct OnnxConnectorImpl : public OnnxConnector {
    const char* c_inputIds = "input_ids";
    const char* c_attentionMask = "attention_mask";
    const char* c_logits = "logits";

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

        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        auto inputShape = std::array<int64_t, 2> { 1, static_cast<int64_t>(tokens.size()) };
        auto idTensor = Ort::Value::CreateTensor<int64_t>(memory_info, const_cast<int64_t*>(tokens.data()), tokens.size(), inputShape.data(), inputShape.size());
        auto maskTensor = Ort::Value::CreateTensor<int64_t>(memory_info, const_cast<int64_t*>(attentionMask.data()), attentionMask.size(), inputShape.data(), inputShape.size());

        auto outputShape = std::array<int64_t, 3> { 1, static_cast<int64_t>(tokens.size()), static_cast<int64_t>(m_tokenIdCount) };
        std::vector<float> logits(tokens.size() * m_tokenIdCount, 0.0f);
        auto outputTensor = Ort::Value::CreateTensor<float>(memory_info, logits.data(), logits.size(), outputShape.data(), outputShape.size());

        auto ioBinding = Ort::IoBinding(m_session);
        ioBinding.BindInput("input_ids", idTensor);
        ioBinding.BindInput("attention_mask", maskTensor);
        ioBinding.BindOutput("logits", outputTensor);

        auto runOptions = Ort::RunOptions();
        m_session.Run(runOptions, ioBinding);

        const auto maxIndex = FindMaxIndex(logits, m_tokenIdCount * (tokens.size() - 1), 32000 * tokens.size());
        return std::make_tuple(maxIndex, logits[m_tokenIdCount * (tokens.size() - 1) + maxIndex]); // TODO: should softmax
    }
    catch (...) { return std::make_tuple(-1LL, 0.0f); }

    std::vector<std::vector<float>> CompareSentences(const std::vector<std::vector<int>>& sentences, int eosId) override try {
        auto lastTime = std::chrono::system_clock::now();
        EnsureInitialized();
        // wprintf(L"Model setup: %lld\n", std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - lastTime).count()); lastTime = std::chrono::system_clock::now();


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

        // wprintf(L"Setup input: %lld\n", std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - lastTime).count()); lastTime = std::chrono::system_clock::now();

        // binding input
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        auto inputShape = std::array<int64_t, 2> { static_cast<int64_t>(sentences.size()), static_cast<int64_t>(maxTokenSize) };
        auto idTensor = Ort::Value::CreateTensor<int64_t>(memory_info, const_cast<int64_t*>(tokenArray.data()), tokenArray.size(), inputShape.data(), inputShape.size());
        auto maskTensor = Ort::Value::CreateTensor<int64_t>(memory_info, const_cast<int64_t*>(attentionMaskArray.data()), attentionMaskArray.size(), inputShape.data(), inputShape.size());

        MemAlignedTensor tokenVector;
        tokenVector.Reserve(sentences.size() * maxTokenSize, m_tokenIdCount);
        auto [outDataPtr, outDataRowCount, outDataColumnCount] = tokenVector.GetBuffer();

        auto outputShape = std::array<int64_t, 3> { static_cast<int64_t>(sentences.size()), static_cast<int64_t>(maxTokenSize), static_cast<int64_t>(m_tokenIdCount) };
        auto outputTensor = Ort::Value::CreateTensor<float>(memory_info, outDataPtr, sentences.size() * maxTokenSize * m_tokenIdCount, outputShape.data(), outputShape.size());

        auto ioBinding = Ort::IoBinding(m_session);
        ioBinding.BindInput(c_inputIds, idTensor);
        ioBinding.BindInput(c_attentionMask, maskTensor);
        ioBinding.BindOutput(c_logits, outputTensor);

        // wprintf(L"Bind in/out: %lld\n", std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - lastTime).count()); lastTime = std::chrono::system_clock::now();

        auto runOptions = Ort::RunOptions();
        m_session.Run(runOptions, ioBinding);

        // wprintf(L"Model exec: %lld\n", std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - lastTime).count()); lastTime = std::chrono::system_clock::now();

        std::vector<std::vector<float>> result;
        for (size_t i = 0; i < sentences.size(); ++i) {
            std::vector<float> probList;
            probList.emplace_back(1.0f);
            for (size_t tokenIndex = 1; tokenIndex < sentences[i].size(); ++tokenIndex) {
                const auto targetVectorOffset = static_cast<int>(i * maxTokenSize * m_tokenIdCount + (tokenIndex - 1) * m_tokenIdCount);
                const auto probability = tokenVector.GetProbabilityInRange(sentences[i][tokenIndex], targetVectorOffset, targetVectorOffset + m_tokenIdCount);
                probList.emplace_back(probability);
            }
            const auto targetVectorOffset = static_cast<int>(i * maxTokenSize * m_tokenIdCount + (sentences[i].size() - 1) * m_tokenIdCount);
            const auto probability = tokenVector.GetProbabilityInRange(eosId, targetVectorOffset, targetVectorOffset + m_tokenIdCount);
            probList.emplace_back(probability);

            result.emplace_back(std::move(probList));
        }

        // wprintf(L"Read output: %lld\n", std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - lastTime).count()); lastTime = std::chrono::system_clock::now();

        return result;
    }
    catch (...) { return std::vector<std::vector<float>>(); }

private:
    void EnsureInitialized() {
        if (!m_session) {
            Ort::Env env;
            Ort::SessionOptions sessionOptions;

            m_session = Ort::Session(env, m_modelFileName.c_str(), sessionOptions);
            m_tokenIdCount = GetTokenIdCount();

            // PrintInputOutput();
        }
    }

    int64_t FindMaxIndex(const std::vector<float>& list, size_t fromIndex, size_t toIndex) {
        size_t resultIndex = fromIndex;
        float maxLogit = list[fromIndex];
        for (size_t i = fromIndex + 1; i < toIndex; ++i) {
            if (maxLogit < list[i]) {
                maxLogit = list[i];
                resultIndex = i;
            }
        }
        return static_cast<int64_t>(resultIndex - fromIndex);
    }

    int64_t GetTokenIdCount()
    {
        Ort::AllocatorWithDefaultOptions alloc;
        const auto logitsStr = std::string(c_logits);

        for (size_t i = 0; i < m_session.GetOutputCount(); ++i) {
            auto outputName = m_session.GetOutputNameAllocated(i, alloc);
            if (logitsStr == outputName.get()) {
                auto outputType = m_session.GetOutputTypeInfo(i);
                auto shapeInfo = outputType.GetTensorTypeAndShapeInfo();
                auto shape = shapeInfo.GetShape();

                if (shape[0] != -1 || shape[1] != -1) {
                    throw std::runtime_error("unexpected shape");
                }
                return shape[2];
            }
        }
        throw std::runtime_error("unexpected shape");
    }

    void PrintInputOutput()
    {
/* result of this function:
    input_ids: Shape:[-1,-1], Type:INT64
    attention_mask: Shape:[-1,-1], Type:INT64
    logits: Shape:[-1,-1,32000], Type:FLOAT
*/

        std::vector<const char*> inputNames;
        std::vector<const char*> outputNames;

        Ort::AllocatorWithDefaultOptions alloc;
        for (size_t i = 0; i < m_session.GetInputCount(); ++i) {
            auto inputName = m_session.GetInputNameAllocated(i, alloc);
            auto inputType = m_session.GetInputTypeInfo(i);
            auto shapeInfo = inputType.GetTensorTypeAndShapeInfo();
            auto shape = shapeInfo.GetShape();
            auto elementType = shapeInfo.GetElementType();

            std::wstringstream ss;
            ss << inputName.get() << L": ";
            ss << L"Shape:" << VectorToText(shape) << L", ";
            ss << L"Type:" << TypeIdToTypeText(elementType);
            wprintf(L"%s\n", ss.str().c_str());
        }
        for (size_t i = 0; i < m_session.GetOutputCount(); ++i) {
            auto outputName = m_session.GetOutputNameAllocated(i, alloc);
            auto outputType = m_session.GetOutputTypeInfo(i);
            auto shapeInfo = outputType.GetTensorTypeAndShapeInfo();
            auto shape = shapeInfo.GetShape();
            auto elementType = shapeInfo.GetElementType();

            std::wstringstream ss;
            ss << outputName.get() << L": ";
            ss << L"Shape:" << VectorToText(shape) << L", ";
            ss << L"Type:" << TypeIdToTypeText(elementType);
            wprintf(L"%s\n", ss.str().c_str());
        }
    }

    std::wstring VectorToText(const std::vector<int64_t>& dim) {
        std::wstringstream ss;
        bool isFirst = true;
        ss << L"[";
        for (const auto x : dim) {
            if (isFirst) {
                isFirst = false;
            } else {
                ss << L",";
            }
            ss << x;
        }
        ss << L"]";
        return ss.str();
    }

    std::wstring TypeIdToTypeText(ONNXTensorElementDataType dataType) {
        std::wstring typeText;
        switch (dataType) {
        default:
            typeText = L"UNDEFINED??"; break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:
            typeText = L"UNDEFINED"; break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:   // maps to c type float
            typeText = L"FLOAT"; break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:   // maps to c type uint8_t
            typeText = L"UINT8"; break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:    // maps to c type int8_t
            typeText = L"INT8"; break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:  // maps to c type uint16_t
            typeText = L"UINT16"; break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:   // maps to c type int16_t
            typeText = L"INT16"; break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:   // maps to c type int32_t
            typeText = L"INT32"; break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:   // maps to c type int64_t
            typeText = L"INT64"; break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:  // maps to c++ type std::string
            typeText = L"STRING"; break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
            typeText = L"BOOL"; break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
            typeText = L"FLOAT16"; break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:      // maps to c type double
            typeText = L"DOUBLE"; break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:      // maps to c type uint32_t
            typeText = L"UINT32"; break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:      // maps to c type uint64_t
            typeText = L"UINT64"; break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:   // complex with float32 real and imaginary components
            typeText = L"COMPLEX64"; break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:  // complex with float64 real and imaginary components
            typeText = L"COMPLEX128"; break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:    // Non-IEEE floating-point format based on IEEE754 single-precision
            typeText = L"BFLOAT16"; break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN:    // Non-IEEE floating-point format based on IEEE754 single-precision
            typeText = L"FLOAT8E4M3FN"; break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FNUZ:  // Non-IEEE floating-point format based on IEEE754 single-precision
            typeText = L"FLOAT8E4M3FNUZ"; break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2:      // Non-IEEE floating-point format based on IEEE754 single-precision
            typeText = L"FLOAT8E5M2"; break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2FNUZ:   // Non-IEEE floating-point format based on IEEE754 single-precision
            typeText = L"FLOAT8E5M2FNUZ"; break;
        }
        return typeText;
    }

private:
    std::wstring m_modelFileName;
    Ort::Session m_session{ nullptr };
    size_t m_tokenIdCount;
};

std::shared_ptr<OnnxConnector> OnnxConnector::CreateInstance() {
    return std::make_shared<OnnxConnectorImpl>();
}


