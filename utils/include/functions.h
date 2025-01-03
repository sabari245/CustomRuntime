#pragma once

#include "types.h"

namespace utility
{

	// ND array creation functions
	template <typename T>
	types::ArrayND<T> createOnes(const std::vector<int> &shape);

	template <typename T>
	types::ArrayND<T> createZeros(const std::vector<int> &shape);

	// ND array accessing functions
	template <typename T>
	T get(const types::ArrayND<T> &array, const std::vector<int> &indices);

	template <typename T>
	types::ArrayND<T> slice(const types::ArrayND<T> &array, int start, int end);

	template <typename T>
	types::ArrayND<T> sliceND(const types::ArrayND<T> &array,
							  const std::vector<int> &start,
							  const std::vector<int> &end);

	// ND array manipulation functions
	template <typename T>
	types::ArrayND<T> reshape(const types::ArrayND<T> &array, const std::vector<int> &newShape);

	template <typename T>
	void reshape_inplace(types::ArrayND<T> &array, const std::vector<int> &newShape);

	template <typename T>
	types::ArrayND<T> addPadding(const types::ArrayND<T> &array, const std::vector<int> &padding);

	template <typename T>
	types::ArrayND<T> transpose(const types::ArrayND<T> &array, const std::vector<int> &perm);

	template <typename T>
	types::ArrayND<T> flatten(const types::ArrayND<T> &array);

	template <typename T>
	void flatten_inplace(types::ArrayND<T> &array);

	template <typename T>
	types::ArrayND<T> add(const types::ArrayND<T> &array1, const types::ArrayND<T> &array2);

	template <typename T>
	types::ArrayND<T> add(const types::ArrayND<T> &array, T scalar);

	template <typename T>
	void add_inplace(types::ArrayND<T> &array1, const types::ArrayND<T> &array2);

	template <typename T>
	types::ArrayND<T> mul(const types::ArrayND<T> &array1, const types::ArrayND<T> &array2);

	template <typename T>
	void mul_inplace(types::ArrayND<T> &array1, const types::ArrayND<T> &array2);

	// Miscilaneous functions
	std::vector<int> getStrideFromShape(const std::vector<int> &shape);

	template <typename T>
	int getOffset(const types::ArrayND<T> &array, const std::vector<int> &indices);

	template <typename T>
	types::ArrayND<T> convertUint8ToType(const types::ArrayND<uint8_t> &array);
}