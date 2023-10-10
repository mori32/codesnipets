#define WIN32_LEAN_AND_MEAN // Exclude rarely-used stuff from Windows headers
#include <windows.h>
#include <cassert>
#include <cfloat>
#include <cstdint>
#include <exception>
#include <tuple>
#include "MemAlignedTensor.h"
#include <immintrin.h>

#pragma optimize("", on)

void MemAlignedTensor::SubstractPosition(const MemAlignedTensor& wpe, int position) {
	assert(wpe.m_column == m_column);
	float* positionVector = wpe.m_body + wpe.m_column * position;
	Subtract(m_body, positionVector, m_column);
}

std::tuple<int64_t, float> MemAlignedTensor::FindToken(const MemAlignedTensor& wte) {
	assert(wte.m_column == m_column);

	const float* wordEmbedding = wte.m_body;
	float simCosVal = -FLT_MAX;
	int simCosIdx = -1;
	auto sumExpVec = _mm256_setzero_ps();

	for (int wordIdx = 0; wordIdx < wte.m_row; wordIdx += 8) {
		__m256 cosValStore;
		for (int i = 0; i < 8; ++i, wordEmbedding += m_column) {
			const auto curCosSim = cosValStore.m256_f32[i] = InnerProduct(m_body, wordEmbedding, m_column);
			if (simCosVal < curCosSim) {
				simCosVal = curCosSim;
				simCosIdx = wordIdx + i;
			}
		}
		const auto v = _mm256_exp_ps(cosValStore);
		sumExpVec = _mm256_add_ps(sumExpVec, v);
	}
	const auto sumExp = HorizontalAdd(sumExpVec);
	const auto prob = static_cast<float>(exp(simCosVal) / sumExp);
	return std::make_tuple(static_cast<int64_t>(simCosIdx), prob);
}

void MemAlignedTensor::Subtract(float* tokenBody, const float* positionVector, int size) {
	for (int i = 0; i < size; i += 32) {
		*reinterpret_cast<__m256*>(tokenBody + 0) = _mm256_sub_ps(*reinterpret_cast<__m256*>(tokenBody + 0), *reinterpret_cast<const __m256*>(positionVector + 0));
		*reinterpret_cast<__m256*>(tokenBody + 8) = _mm256_sub_ps(*reinterpret_cast<__m256*>(tokenBody + 8), *reinterpret_cast<const __m256*>(positionVector + 8));
		*reinterpret_cast<__m256*>(tokenBody + 16) = _mm256_sub_ps(*reinterpret_cast<__m256*>(tokenBody + 16), *reinterpret_cast<const __m256*>(positionVector + 16));
		*reinterpret_cast<__m256*>(tokenBody + 24) = _mm256_sub_ps(*reinterpret_cast<__m256*>(tokenBody + 24), *reinterpret_cast<const __m256*>(positionVector + 24));
		tokenBody += 32;
		positionVector += 32;
	}
}

float MemAlignedTensor::InnerProduct(float* tokenBody, const float* wordEmbed, int size) {
	auto sumVector = _mm256_setzero_ps();
	for (int i = 0; i < size; i += 32) {
		const auto v1 = _mm256_mul_ps(*reinterpret_cast<__m256*>(tokenBody + 0), *reinterpret_cast<const __m256*>(wordEmbed + 0));
		const auto v2 = _mm256_mul_ps(*reinterpret_cast<__m256*>(tokenBody + 8), *reinterpret_cast<const __m256*>(wordEmbed + 8));
		const auto v3 = _mm256_mul_ps(*reinterpret_cast<__m256*>(tokenBody + 16), *reinterpret_cast<const __m256*>(wordEmbed + 16));
		const auto v4 = _mm256_mul_ps(*reinterpret_cast<__m256*>(tokenBody + 24), *reinterpret_cast<const __m256*>(wordEmbed + 24));
		const auto v12 = _mm256_add_ps(v1, v2);
		const auto v34 = _mm256_add_ps(v3, v4);
		const auto v1234 = _mm256_add_ps(v12, v34);
		sumVector = _mm256_add_ps(sumVector, v1234);
		tokenBody += 32;
		wordEmbed += 32;
	}
	return HorizontalAdd(sumVector);
}

std::tuple<int64_t, float> MemAlignedTensor::GetIndexFromLogits() {
#ifdef _M_X64
	int cosSimIdx = -1;
	float cosSimScore = -FLT_MAX;

	__m256 tmpBuf[4];
	auto tmpSumVec = _mm256_setzero_ps();
	for (auto i = 0; i < m_column; i += 32) {
		const auto v1 = tmpBuf[0] = _mm256_exp_ps(*(reinterpret_cast<__m256*>(m_body + i + 0)));
		const auto v2 = tmpBuf[1] = _mm256_exp_ps(*(reinterpret_cast<__m256*>(m_body + i + 8)));
		const auto v3 = tmpBuf[2] = _mm256_exp_ps(*(reinterpret_cast<__m256*>(m_body + i + 16)));
		const auto v4 = tmpBuf[3] = _mm256_exp_ps(*(reinterpret_cast<__m256*>(m_body + i + 24)));
		const auto v12 = _mm256_add_ps(v1, v2);
		const auto v34 = _mm256_add_ps(v3, v4);
		const auto v1234 = _mm256_add_ps(v12, v34);
		tmpSumVec = _mm256_add_ps(tmpSumVec, v1234);

		const auto maxV12 = _mm256_max_ps(v1, v2);
		const auto maxV34 = _mm256_max_ps(v3, v4);
		const auto maxV1234 = _mm256_max_ps(maxV12, maxV34);

		if (cosSimScore < HorizontalMax(maxV1234)) {
			for (int j = 0; j < 8 * 4; ++j) {
				if (cosSimScore < tmpBuf[0].m256_f32[j]) {
					cosSimScore = tmpBuf[0].m256_f32[j];
					cosSimIdx = i + j;
				}
			}
		}
	}
	const auto sumf = HorizontalAdd(tmpSumVec);

	return std::make_tuple(cosSimIdx, cosSimScore / sumf);
#else
	assert(false); // not impl
#endif
}

float MemAlignedTensor::GetProbability(int tokenIndex) {
	return GetProbabilityInRange(tokenIndex, 0, m_column);
}

float MemAlignedTensor::GetProbabilityInRange(int tokenIndex, int fromIdx, int toIdx) {
	auto tmpSumVec = _mm256_setzero_ps();
	for (auto i = fromIdx; i < toIdx; i += 32) {
		const auto v1 = _mm256_exp_ps(*(reinterpret_cast<__m256*>(m_body + i + 0)));
		const auto v2 = _mm256_exp_ps(*(reinterpret_cast<__m256*>(m_body + i + 8)));
		const auto v3 = _mm256_exp_ps(*(reinterpret_cast<__m256*>(m_body + i + 16)));
		const auto v4 = _mm256_exp_ps(*(reinterpret_cast<__m256*>(m_body + i + 24)));
		const auto v12 = _mm256_add_ps(v1, v2);
		const auto v34 = _mm256_add_ps(v3, v4);
		const auto v1234 = _mm256_add_ps(v12, v34);
		tmpSumVec = _mm256_add_ps(tmpSumVec, v1234);
	}

	const auto sumf = HorizontalAdd(tmpSumVec);
	return expf(m_body[fromIdx + tokenIndex]) / sumf;
}

float MemAlignedTensor::HorizontalMax(const __m256& x) {
	const __m128 hiQuad = _mm256_extractf128_ps(x, 1);			// hiQuad = ( x7, x6, x5, x4 )
	const __m128 loQuad = _mm256_castps256_ps128(x);        	// loQuad = ( x3, x2, x1, x0 )
	const __m128 maxQuad = _mm_max_ps(loQuad, hiQuad);      	// sumQuad = ( x3 + x7, x2 + x6, x1 + x5, x0 + x4 )
	const __m128 loDual = maxQuad;                          	// loDual = ( -, -, x1 + x5, x0 + x4 )
	const __m128 hiDual = _mm_movehl_ps(maxQuad, maxQuad);  	// hiDual = ( -, -, x3 + x7, x2 + x6 )
	const __m128 maxDual = _mm_max_ps(loDual, hiDual);      	// sumDual = ( -, -, x1 + x3 + x5 + x7, x0 + x2 + x4 + x6 )
	const __m128 lo = maxDual;                              	// lo = ( -, -, -, x0 + x2 + x4 + x6 )
	const __m128 hi = _mm_shuffle_ps(maxDual, maxDual, 0x1);	// hi = ( -, -, -, x1 + x3 + x5 + x7 )
	const __m128 maxS = _mm_max_ps(lo, hi);						// sum = ( -, -, -, x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 )	}
	return _mm_cvtss_f32(maxS);
}

float MemAlignedTensor::HorizontalAdd(const __m256& x) {
	const __m128 hiQuad = _mm256_extractf128_ps(x, 1);			// hiQuad = ( x7, x6, x5, x4 )
	const __m128 loQuad = _mm256_castps256_ps128(x);			// loQuad = ( x3, x2, x1, x0 )
	const __m128 sumQuad = _mm_add_ps(loQuad, hiQuad);			// sumQuad = ( x3 + x7, x2 + x6, x1 + x5, x0 + x4 )
	const __m128 loDual = sumQuad;                              // loDual = ( -, -, x1 + x5, x0 + x4 )
	const __m128 hiDual = _mm_movehl_ps(sumQuad, sumQuad);      // hiDual = ( -, -, x3 + x7, x2 + x6 )
	const __m128 sumDual = _mm_add_ps(loDual, hiDual);			// sumDual = ( -, -, x1 + x3 + x5 + x7, x0 + x2 + x4 + x6 )
	const __m128 lo = sumDual;                                  // lo = ( -, -, -, x0 + x2 + x4 + x6 )
	const __m128 hi = _mm_shuffle_ps(sumDual, sumDual, 0x1);	// hi = ( -, -, -, x1 + x3 + x5 + x7 )
	const __m128 sum = _mm_add_ss(lo, hi);						// sum = ( -, -, -, x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 )	}
	return _mm_cvtss_f32(sum);
}
