#pragma once
#include <windows.h>
#include <exception>
#include <string>

inline std::string ToUtf8(std::wstring_view src) {
	const auto requiredSize = WideCharToMultiByte(CP_UTF8, 0, src.data(), static_cast<int>(src.length()), nullptr, 0, nullptr, nullptr);
	std::string utf8(requiredSize, ' ');
	WideCharToMultiByte(CP_UTF8, 0, src.data(), static_cast<int>(src.length()), utf8.data(), requiredSize, nullptr, nullptr);
	return utf8;
}

inline std::wstring ToUtf16(std::string_view src) {
	const auto requiredSize = MultiByteToWideChar(CP_UTF8, 0, src.data(), static_cast<int>(src.length()), nullptr, 0);
	std::wstring utf16(requiredSize, L' ');
	MultiByteToWideChar(CP_UTF8, 0, src.data(), static_cast<int>(src.length()), utf16.data(), requiredSize);
	return utf16;
}
