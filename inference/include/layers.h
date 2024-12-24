#pragma once

#include <vector>

namespace layers {

	template <typename T>
	struct ArrayND {
		std::vector<int> shape;
		std::vector<int> stride;
		std::vector<T> data;
	};

	class Layers {
	public:
		template <typename T>
		void transpose(
			const ArrayND<T>& in,
			const std::vector<int>& permutation,
			ArrayND<T>& out
		);

		template <typename T>
		void conv2d(
			const ArrayND<T>& in,
			const ArrayND<T>& weights,
			const ArrayND<T>& bias,
			const std::vector<int>& padding,
			const std::vector<int>& stride,
			const std::vector<int>& kernel_shape,
			ArrayND<T>& out
		);

	};
}