#include <iostream>
#include <io.h>
#include <fcntl.h>
#include <stdio.h>
#include "tokenizer.h"
#include "onnxConnector.h"

#if 0 // old test code
void TestSentencePiece() {
	auto&& tokenizer = Tokenizer::CreateInstance();
	tokenizer->Load(L"../my_onnx_gpt/spiece.model");

	const auto tokenVector = tokenizer->Encode64(L"こんにちは");

	for (const auto& token : tokenVector) {
		printf("%lld, ", token);
	}
	printf("\n");

	const auto decodedText = tokenizer->Decode64(&tokenVector[0], tokenVector.size());

	wprintf(L"%s\n", decodedText.c_str());
}

void TestOnnxModel() {
	auto&& tokenizer = Tokenizer::CreateInstance();
	tokenizer->Load(L"../my_onnx_gpt/spiece.model");

	auto&& onnx = OnnxConnector::CreateInstance();
	onnx->Initialize(L"../my_onnx_gpt/decoder_model.onnx");

	const auto tokenVector = tokenizer->Encode64(L"こんにちは");

	auto [nextToken, nextProb] = onnx->GetPrediction(tokenVector);

	const auto decodedText = tokenizer->Decode64(&nextToken, 1);
}

void TestLongPrediction(std::wstring_view sourceText) {
	wprintf(L"%s => ", sourceText.data());

	auto&& tokenizer = Tokenizer::CreateInstance();
	tokenizer->Load(L"../my_onnx_gpt/spiece.model");

	auto&& onnx = OnnxConnector::CreateInstance();
	onnx->Initialize(L"../my_onnx_gpt/decoder_model.onnx");

	std::wstring currentText(sourceText);

	// get next 10 tokens by greedy algorithm.
	for (int i = 0; i < 10; ++i) {
		auto tokenVector = tokenizer->Encode64(currentText);

		auto [nextToken, nextProb] = onnx->GetPrediction(tokenVector);

		tokenVector.push_back(nextToken);
		currentText = tokenizer->Decode64(&tokenVector[0], tokenVector.size());
	}

	wprintf(L"%s\n", currentText.c_str());
}
#endif // old test code

void CompareSentences(const std::vector<const wchar_t*> sentences) {
	auto&& tokenizer = Tokenizer::CreateInstance();
	tokenizer->Load(L"../my_onnx_gpt/spiece.model");

	auto&& onnx = OnnxConnector::CreateInstance();
	onnx->Initialize(L"../my_onnx_gpt/decoder_model.onnx");

	std::vector<std::vector<int>> tokensList;
	for (const auto sentence : sentences) {
		auto&& encodedSentence = tokenizer->Encode(sentence);
		tokensList.emplace_back(encodedSentence);
	}

	const auto& resultMatrix = onnx->CompareSentences(tokensList, tokenizer->eos_id());

	for (size_t i = 0; i < resultMatrix.size(); ++i) {
		wprintf(L"%s: ", sentences[i]);
		const auto& tokens = tokensList[i];
		for (const auto token : tokens) wprintf(L"%d,", token);
		wprintf(L": ");
		const auto& probList = resultMatrix[i];
		float logProb = 0.0f;
		for (const auto prob : probList) {
			wprintf(L"%f,", prob);
			logProb += logf(prob);
		}
		wprintf(L": %f\n", logProb);
	}
}



int main()
{
	(void)_setmode(_fileno(stdout), _O_U16TEXT);

	CompareSentences({
			L"庭で犬を飼う",
			L"庭で犬を買う",
			L"庭で犬をかう"
		});

	CompareSentences({
			L"店で犬を飼う",
			L"店で犬を買う",
			L"店で犬をかう"
		});

	CompareSentences({
			L"登校時間が、いつもよりとても速い",
			L"登校時間が、いつもよりとても早い",
		});

	CompareSentences({
			L"彼の足は、いつもよりとても速い",
			L"彼の足は、いつもよりとても早い",
		});

	return 0;
}