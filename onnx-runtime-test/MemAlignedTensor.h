// (C) 2023 millimoji@gmail.com
#pragma once

#include <immintrin.h>

class MemAlignedTensor
{
private:
	static constexpr uint32_t alignmentSize = 64;

public:
	MemAlignedTensor() = default;
	MemAlignedTensor(const MemAlignedTensor&) = delete;
	MemAlignedTensor(MemAlignedTensor&& src) noexcept {
		m_row = src.m_row; m_column = src.m_column; m_body = src.m_body; src.m_body = nullptr;
	}
	~MemAlignedTensor() { _aligned_free(m_body); }

	MemAlignedTensor& operator = (const MemAlignedTensor&) = delete;
	MemAlignedTensor& operator = (MemAlignedTensor&& src) noexcept {
		m_row = src.m_row; m_column = src.m_column; m_body = src.m_body; src.m_body = nullptr;
		return *this;
	}

	MemAlignedTensor(int64_t row, int64_t column, const float* src) {
		Copy(row, column, src);
	}
	std::tuple<float*, int, int> GetBuffer() {
		return std::make_tuple(m_body, m_row, m_column);
	}
	float* Reserve(int64_t row, int64_t column) {
		if ((row * column) != (m_row * m_column)) {
			void* pv = _aligned_malloc(row * column * sizeof(float), alignmentSize);
			if (pv == nullptr) throw std::bad_alloc();
			_aligned_free(m_body);
			m_body = reinterpret_cast<float*>(pv);
		}
		m_row = static_cast<int>(row); m_column = static_cast<int>(column);
		return m_body;
	}
	void Copy(int64_t row, int64_t column, const float* src) {
		Reserve(row, column);
		if (src != nullptr) {
			memcpy(m_body, src, row * column * sizeof(float));
		}
	}

	// intel avx code
	void SubstractPosition(const MemAlignedTensor& wpe, int position);
	std::tuple<int64_t, float> FindToken(const MemAlignedTensor& wte);
	std::tuple<int64_t, float> GetIndexFromLogits();
	float GetProbability(int tokenIndex);
	float GetProbabilityInRange(int tokenIndex, int fromIdx, int toIdx);
	static void Subtract(float* tokenBody, const float* positionVector, int size);
	static float InnerProduct(float* tokenBody, const float* wordEmbed, int size);
	static float HorizontalMax(const __m256& x);
	static float HorizontalAdd(const __m256& x);

private:
	int m_row = 0;
	int m_column = 0;
	float* m_body = nullptr;
};
