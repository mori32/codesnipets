#pragma once
#include <memory>
#include <string>
#include <string_view>
#include <vector>

struct Tokenizer
{
	virtual void Load(std::wstring_view fileName) = 0;
	virtual std::vector<int64_t> Encode(std::wstring_view source) = 0;
	virtual std::wstring Decode(const int64_t* tokenPtr, size_t tokenLen) = 0;

	virtual ~Tokenizer() {};
	static std::shared_ptr<Tokenizer> CreateInstance();
};


