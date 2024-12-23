#pragma once

#include <vector>

namespace layers {

	class Layers {
	public:
		template <typename T>
		void transpose(
			const std::vector<int>& shape,
			const std::vector<int>& stride,
			const std::vector<T>& datax,
			const std::vector<int>& permutation,
			std::vector<int>& outShape,
			std::vector<int>& outStride,
			std::vector<T>& outDatax
		);
	};
}