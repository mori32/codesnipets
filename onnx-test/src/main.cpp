#include <iostream>
#include <io.h>
#include <fcntl.h>
#include <stdio.h>
#include "tokenizer.h"
#include "onnxConnector.h"

//const std::wstring modelDir = L"../../onnx-models/rinna-gpt2-xsmall/";
//const std::wstring modelDir = L"../../onnx-models/rinna-japanese-gpt2-small/";
const std::wstring modelDir = L"../../onnx-models/rinna-neox-3.6b/";

void TestSentencePiece() {
	auto&& tokenizer = Tokenizer::CreateInstance();
	tokenizer->Load(L"../my_onnx_gpt/spiece.model");

	const auto tokenVector = tokenizer->Encode(L"こんにちは");

	for (const auto& token : tokenVector) {
		printf("%ld, ", token);
	}
	printf("\n");

	const auto decodedText = tokenizer->Decode(&tokenVector[0], tokenVector.size());

	wprintf(L"%s\n", decodedText.c_str());
}

void TestOnnxModel() {
	auto&& tokenizer = Tokenizer::CreateInstance();
	tokenizer->Load(L"../my_onnx_gpt/spiece.model");

	auto&& onnx = OnnxConnector::CreateInstance();
	onnx->Initialize(L"../my_onnx_gpt/decoder_model.onnx");

	const auto tokenVector = tokenizer->Encode(L"こんにちは");

	auto [nextToken, nextProb] = onnx->GetPrediction(tokenVector);

	const auto decodedText = tokenizer->Decode(&nextToken, 1);
}

void TestLongPrediction(std::wstring_view sourceText) {
	wprintf(L"%s => ", sourceText.data());

	auto&& tokenizer = Tokenizer::CreateInstance();
	tokenizer->Load((modelDir + L"spiece.model").c_str());

	auto&& onnx = OnnxConnector::CreateInstance();
	onnx->Initialize((modelDir + L"decoder_model.onnx").c_str());

	std::wstring currentText(sourceText);

	// get next 10 tokens by greedy algorithm.
	for (int i = 0; i < 10; ++i) {
		auto tokenVector = tokenizer->Encode(currentText);

		auto [nextToken, nextProb] = onnx->GetPrediction(tokenVector);

		tokenVector.push_back(nextToken);
		currentText = tokenizer->Decode(&tokenVector[0], tokenVector.size());
	}

	wprintf(L"%s\n", currentText.c_str());
}


int main()
{
	(void)_setmode(_fileno(stdout), _O_U16TEXT);

	TestLongPrediction(L"昔々あるところに");
	TestLongPrediction(L"このたびは誠に");
	TestLongPrediction(L"本日はお日柄もよく");
	return 0;
}