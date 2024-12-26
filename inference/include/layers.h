#pragma once

#include <vector>
#include "types.h"
#include "functions.h"

namespace layers {

	class Layers {
	private:
		template <typename T>
		types::ArrayND<T> convolutionOperation(
			const types::ArrayND<T>& data,
			const types::ArrayND<T>& kernel,
			const std::vector<int>& padding,
			const std::vector<int>& stride
		);

	public:
		template <typename T>
		types::ArrayND<T> transpose(
			const types::ArrayND<T>& in,
			const std::vector<int>& permutation
		);

		template <typename T>
		types::ArrayND<T> conv2d(
			const types::ArrayND<T>& in,
			const types::ArrayND<T>& weights,
			const types::ArrayND<T>& bias,
			const std::vector<int>& padding,
			const std::vector<int>& stride,
			const std::vector<int>& kernel_shape
		);

		template <typename T>
		types::ArrayND<T> Relu(
			const types::ArrayND<T>& in
		);

		template <typename T>
		types::ArrayND<T> maxPool(
			const types::ArrayND<T>& in,
			const std::vector<int>& kernel_shape,
			const std::vector<int>& stride
		);

		template <typename T>
		types::ArrayND<T> Reshape(
			const types::ArrayND<T>& in,
			const types::ArrayND<T>& shape
		);

		template <typename T>
		types::ArrayND<T> MatMul(
			const types::ArrayND<T>& in1,
			const types::ArrayND<T>& in2
		);

		template <typename T>
		types::ArrayND<T> Add(
			const types::ArrayND<T>& in1,
			const types::ArrayND<T>& in2
		);

		template <typename T>
		types::ArrayND<T> Mul(
			const types::ArrayND<T>& in1,
			const types::ArrayND<T>& in2
		);

		template <typename T>
		types::ArrayND<T> Softmax(
			const types::ArrayND<T>& in
		);
	};
}