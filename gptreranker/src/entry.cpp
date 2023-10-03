#include <Windows.h>
#include <iostream>
#include <io.h>
#include <fcntl.h>
#include <stdio.h>
#include "tokenizer.h"
#include "onnxConnector.h"

HMODULE GetThisModuleHandle() {
	HMODULE hModule = {};
	if (!GetModuleHandleEx(
		GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
		reinterpret_cast<LPCWSTR>(GetThisModuleHandle), &hModule)) {
		return nullptr;
	}
	return hModule;
}

std::wstring GetModuleFilePath(HMODULE hModule) {
	wchar_t moduleName[MAX_PATH] = {};
	if (GetModuleFileName(hModule, moduleName, ARRAYSIZE(moduleName)) == 0) {
		return std::wstring(L"");
	}
	return std::wstring(moduleName);
}

std::wstring GetPathPart(const std::wstring& filePath) {
	wchar_t fullPath[MAX_PATH] = {};
	wchar_t* filePart;
	if (GetFullPathName(filePath.c_str(), ARRAYSIZE(fullPath), fullPath, &filePart) == 0) {
		return std::wstring(L"");
	}
	return std::wstring(fullPath, filePart - fullPath);
}

std::wstring GetThisModuleDirectory() {
	return GetPathPart(GetModuleFilePath(GetThisModuleHandle()));
}


std::tuple<std::shared_ptr<Tokenizer>, std::shared_ptr<OnnxConnector>> EnsureInitialized() {
	const auto& thisModuleDir = GetThisModuleDirectory();

	static std::shared_ptr<Tokenizer> tokenizer;
	if (!tokenizer) {
		tokenizer = Tokenizer::CreateInstance();
		const auto& modelPath = thisModuleDir + L"spiece.model";
		tokenizer->Load(modelPath.c_str());
	}

	static std::shared_ptr<OnnxConnector> onnx;
	if (!onnx) {
		onnx = OnnxConnector::CreateInstance();
		const auto& modelPath = thisModuleDir + L"decoder_model.onnx";
		onnx->Initialize(modelPath.c_str());
	}

	return std::make_tuple(tokenizer, onnx);
}

extern "C" __declspec(dllexport)
int WINAPI EvaluateSentences(const char** sentences, float* scores, int sentenceCount)
{
	const auto [tokenizer, onnx] = EnsureInitialized();

	std::vector<std::vector<int>> tokensList;
	for (int i = 0; i < sentenceCount; ++i) {
		auto&& encodedSentence = tokenizer->Encode(sentences[i]);
		tokensList.emplace_back(encodedSentence);
	}

	onnx->CompareSentenceDiffs(tokensList, tokenizer->eos_id(), scores);

	return 0;
}

extern "C" __declspec(dllexport)
void WINAPI TestFunction()
{
	(void)_setmode(_fileno(stdout), _O_U16TEXT);

	float probBuf[3];
	{
		const char* input[] = {
				(const char*)u8"庭で犬を飼う",
				(const char*)u8"庭で犬を買う",
				(const char*)u8"庭で犬をかう"
		};
		EvaluateSentences(input, probBuf, 3);
	}

	{
		const char* input[] = {
				(const char*)u8"店で犬を飼う",
				(const char*)u8"店で犬を買う",
				(const char*)u8"店で犬をかう"
		};
		EvaluateSentences(input, probBuf, 3);
	}

	{
		const char* input[] = {
			(const char*)u8"登校時間が、いつもよりとても速い",
			(const char*)u8"登校時間が、いつもよりとても早い",
		};
		EvaluateSentences(input, probBuf, 2);
	}

	{
		const char* input[] = {
			(const char*)u8"彼の足は、いつもよりとても速い",
			(const char*)u8"彼の足は、いつもよりとても早い",
		};
		EvaluateSentences(input, probBuf, 2);
	}
}
