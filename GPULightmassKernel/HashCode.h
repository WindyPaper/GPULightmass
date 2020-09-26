#pragma once

#include <algorithm>

struct HashCode
{
	static const uint Prime1 = 2654435761U;
	static const uint Prime2 = 2246822519U;
	static const uint Prime3 = 3266489917U;
	static const uint Prime4 = 668265263U;
	static const uint Prime5 = 374761393U;

	static const uint s_seed = 10086U;

	static uint MixEmptyState()
	{
		return s_seed + Prime5;
	}

	static uint MixFinal(uint hash)
	{
		hash ^= hash >> 15;
		hash *= Prime2;
		hash ^= hash >> 13;
		hash *= Prime3;
		hash ^= hash >> 16;
		return hash;
	}

	static int leftRotate(unsigned int n, unsigned int d) { //rotate n by d bits
		return (n << d) | (n >> (32 - d));
	}

	static uint QueueRound(uint hash, uint queuedValue)
	{
		return leftRotate(hash + queuedValue * Prime3, 17) * Prime4;
	}

	static int ToHashCode(int value1, int value2, int value3)
	{
		uint hc1 = std::hash<int>()(value1);
		uint hc2 = std::hash<int>()(value2);
		uint hc3 = std::hash<int>()(value3);

		uint hash = MixEmptyState();
		hash += 12;

		hash = QueueRound(hash, hc1);
		hash = QueueRound(hash, hc2);
		hash = QueueRound(hash, hc3);

		hash = MixFinal(hash);
		return (int)hash;
	}
};

int mix_int(int a, int b, int c)
{
	a = a - b;  a = a - c;  a = a ^ ((unsigned int)c >> 13);
	b = b - c;  b = b - a;  b = b ^ (a << 8);
	c = c - a;  c = c - b;  c = c ^ ((unsigned int)b >> 13);
	a = a - b;  a = a - c;  a = a ^ ((unsigned int)c >> 12);
	b = b - c;  b = b - a;  b = b ^ (a << 16);
	c = c - a;  c = c - b;  c = c ^ ((unsigned int)b >> 5);
	a = a - b;  a = a - c;  a = a ^ ((unsigned int)c >> 3);
	b = b - c;  b = b - a;  b = b ^ (a << 10);
	c = c - a;  c = c - b;  c = c ^ ((unsigned int)b >> 15);
	return c;
}

int F3ToIntKey(const float3& f)
{
	int error_unit = 1;

	int x = f.x * error_unit;
	int y = f.y * error_unit;
	int z = f.z * error_unit;

	//return HashCode::ToHashCode(x, y, z);
	return mix_int(x, y, z);
}

inline uint64_t mortonEncode_for(unsigned int x, unsigned int y, unsigned int z) {
	uint64_t answer = 0;
	for (uint64_t i = 0; i < (sizeof(uint64_t) * CHAR_BIT) / 3; ++i) {
		answer |= ((x & ((uint64_t)1 << i)) << 2 * i) | ((y & ((uint64_t)1 << i)) << (2 * i + 1)) | ((z & ((uint64_t)1 << i)) << (2 * i + 2));
	}
	return answer;
}

uint64_t F3ToLongKey(const float3& f)
{
	int error_unit = 1;

	int x = f.x * error_unit;
	int y = f.y * error_unit;
	int z = f.z * error_unit;

	//return HashCode::ToHashCode(x, y, z);
	return mortonEncode_for(x, y, z);
}