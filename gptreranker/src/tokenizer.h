#pragma once
#include <memory>
#include <string>
#include <string_view>
#include <vector>

struct Tokenizer
{
	virtual void Load(std::wstring_view fileName) = 0;
	virtual std::vector<int> Encode(std::string_view source) = 0;
	virtual std::vector<int> Encode(std::wstring_view source) = 0;
	virtual std::vector<int64_t> Encode64(std::wstring_view source) = 0;
	virtual std::wstring Decode64(const int64_t* tokenPtr, size_t tokenLen) = 0;

	virtual int bos_id() = 0;
	virtual int eos_id() = 0;

	virtual ~Tokenizer() {};
	static std::shared_ptr<Tokenizer> CreateInstance();
};
