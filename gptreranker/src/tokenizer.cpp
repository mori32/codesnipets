#include <Windows.h>
#include "miscUtils.h"
#include "tokenizer.h"
#include <sentencepiece_processor.h>

struct TokenizerImpl : public Tokenizer
{
	void Load(std::wstring_view fileName) override {
		const auto& utf8FileName = ToUtf8(fileName);
		m_processor.reset(new sentencepiece::SentencePieceProcessor());
		const auto& result = m_processor->Load(utf8FileName);
		if (!result.ok()) {
			throw std::exception("model loading failed.");
		}
	}

	std::vector<int> Encode(std::string_view source) override {
		if (!m_processor) {
			throw std::exception("processor is not loaded");
		}

		std::vector<int> tokenVector;
		const auto& result = m_processor->Encode(source, &tokenVector);
		if (!result.ok()) throw std::exception("failed to encode");

		return tokenVector;
	}

	std::vector<int> Encode(std::wstring_view source) override {
		if (!m_processor) {
			throw std::exception("processor is not loaded");
		}

		const auto& utf8Text = ToUtf8(source);

		std::vector<int> tokenVector;
		const auto& result = m_processor->Encode(utf8Text, &tokenVector);
		if (!result.ok()) throw std::exception("failed to encode");

		return tokenVector;
	}

	std::vector<int64_t> Encode64(std::wstring_view source) override {
		if (!m_processor) {
			throw std::exception("processor is not loaded");
		}

		const auto& utf8Text = ToUtf8(source);

		std::vector<int> tokenVector;
		const auto& result = m_processor->Encode(utf8Text, &tokenVector);
		if (!result.ok()) throw std::exception("failed to encode");

		std::vector<int64_t> resultVector(tokenVector.size(), 0LL);
		for (auto i = 0ULL; i < tokenVector.size(); ++i) {
			resultVector[i] = tokenVector[i];
		}
		return resultVector;
	}

	std::wstring Decode64(const int64_t* tokenPtr, size_t tokenLen) override {
		if (!m_processor) {
			throw std::exception("processor is not loaded");
		}

		std::vector<int> tokensInt(tokenLen, 0);
		for (auto i = 0U; i < tokenLen; ++i) {
			tokensInt[i] = static_cast<int>(tokenPtr[i]);
		}

		std::string decodedText;
		const auto& result = m_processor->Decode(tokensInt, &decodedText);

		return ToUtf16(decodedText);
	}

	int bos_id() override { return m_processor->bos_id(); }
	int eos_id() override { return m_processor->eos_id(); }

private:
	std::unique_ptr<sentencepiece::SentencePieceProcessor> m_processor;
};

std::shared_ptr<Tokenizer> Tokenizer::CreateInstance() {
	return std::make_shared<TokenizerImpl>();
}


