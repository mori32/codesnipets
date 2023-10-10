#include <iostream>
#include <io.h>
#include <fcntl.h>
#include <stdio.h>
#include "tokenizer.h"
#include "onnxConnector.h"

#pragma comment(lib, "onnxruntime.lib")

//const std::wstring modelDir = L"../onnx-models/rinna-gpt2-xsmall/";
//const std::wstring modelDir = L"../onnx-models/rinna-japanese-gpt2-small/";
const std::wstring modelDir = L"../onnx-models/rinna-neox-3.6b/";

void TestSentencePiece() {
	auto&& tokenizer = Tokenizer::CreateInstance();
	tokenizer->Load((modelDir + L"spiece.model").c_str());

	const auto tokenVector = tokenizer->Encode64(L"こんにちは");

	for (const auto& token : tokenVector) {
		wprintf(L"%ld, ", token);
	}
	wprintf(L"\n");

	const auto decodedText = tokenizer->Decode(&tokenVector[0], tokenVector.size());

	wprintf(L"%s\n", decodedText.c_str());
}

void TestOnnxModel() {
	auto&& tokenizer = Tokenizer::CreateInstance();
	tokenizer->Load((modelDir + L"spiece.model").c_str());

	auto&& onnx = OnnxConnector::CreateInstance();
	onnx->Initialize((modelDir + L"decoder_model.onnx").c_str());

	const auto tokenVector = tokenizer->Encode64(L"こんにちは");

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
		auto tokenVector = tokenizer->Encode64(currentText);

		auto [nextToken, nextProb] = onnx->GetPrediction(tokenVector);

		tokenVector.push_back(nextToken);
		currentText = tokenizer->Decode(&tokenVector[0], tokenVector.size());
	}

	wprintf(L"%s\n", currentText.c_str());
}

void CompareSentences(const std::vector<const wchar_t*> sentences) {
	auto&& tokenizer = Tokenizer::CreateInstance();
	tokenizer->Load((modelDir + L"spiece.model").c_str());

	static std::shared_ptr<OnnxConnector> onnx;
	if (!onnx) {
		onnx = OnnxConnector::CreateInstance();
		onnx->Initialize((modelDir + L"decoder_model.onnx").c_str());
	}

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

#if 0
	TestSentencePiece();
#endif
#if 0
	TestOnnxModel();
#endif
#if 0
	TestLongPrediction(L"昔々あるところに");
	TestLongPrediction(L"このたびは誠に");
	TestLongPrediction(L"本日はお日柄もよく");
#endif

	CompareSentences({
			L"庭で犬を飼う",
			L"庭で犬を買う",
			L"庭で犬をかう"
		});

	CompareSentences({
			L"昨日犬を買った",
			L"昨日犬を飼った",
			L"昨日犬をかった",
			L"昨日犬を勝った",
		});

	CompareSentences({
			L"昨日から犬を買った",
			L"昨日から犬を飼った",
			L"昨日から犬をかった",
			L"昨日から犬を勝った",
		});

	CompareSentences({
			L"私の姉の名前は陽子です。先日、姉の陽子",
			L"私の姉の名前は陽子です。先日、姉の葉子",
			L"私の姉の名前は陽子です。先日、姉の洋子",
		});

	CompareSentences({
			L"私の姉の名前は葉子です。先日、姉の陽子",
			L"私の姉の名前は葉子です。先日、姉の葉子",
			L"私の姉の名前は葉子です。先日、姉の洋子",
		});

	CompareSentences({
			L"私の姉の名前は陽子で、いとこの名前は葉子です。先日、姉の陽子",
			L"私の姉の名前は陽子で、いとこの名前は葉子です。先日、姉の葉子",
		});

	CompareSentences({
			L"私の姉の名前は陽子で、いとこの名前は葉子です。先日、いとこの陽子",
			L"私の姉の名前は陽子で、いとこの名前は葉子です。先日、いとこの葉子",
		});

	return 0;
}