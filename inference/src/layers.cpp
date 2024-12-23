#include <vector>
#include <algorithm>
#include <numeric>
#include <stdexcept>

#include "layers.h"

namespace layers {

    template <typename T>
    void Layers::transpose(
        const std::vector<int>& shape,
        const std::vector<int>& stride,
        const std::vector<T>& datax,
        const std::vector<int>& permutation,
        std::vector<int>& outShape,
        std::vector<int>& outStride,
        std::vector<T>& outDatax
    ) {
        // Validate input
        if (shape.size() != permutation.size()) {
            throw std::invalid_argument("Permutation size must match matrix dimensions");
        }

        // Rearrange shape according to permutation
        outShape.resize(shape.size());
        for (size_t i = 0; i < permutation.size(); i++) {
            outShape[i] = shape[permutation[i]];
        }

        // Calculate new strides
        outStride.resize(shape.size());
        int current_stride = 1;
        for (int i = outShape.size() - 1; i >= 0; i--) {
            outStride[i] = current_stride;
            current_stride *= outShape[i];
        }

        // Initialize output data
        int total_size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
        outDatax.resize(total_size);

        // Create index arrays for iteration
        std::vector<int> current_idx(shape.size(), 0);

        // Iterate through all elements
        for (int flat_idx = 0; flat_idx < total_size; flat_idx++) {
            // Calculate source index
            int src_idx = 0;
            for (size_t dim = 0; dim < shape.size(); dim++) {
                src_idx += current_idx[dim] * stride[dim];
            }

            // Calculate destination index
            int dst_idx = 0;
            for (size_t dim = 0; dim < outShape.size(); dim++) {
                dst_idx += current_idx[permutation[dim]] * outStride[dim];
            }

            // Copy data
            outDatax[dst_idx] = datax[src_idx];

            // Update indices
            for (int dim = shape.size() - 1; dim >= 0; dim--) {
                current_idx[dim]++;
                if (current_idx[dim] < shape[dim]) {
                    break;
                }
                current_idx[dim] = 0;
            }
        }
    }

    template void Layers::transpose<int>(const std::vector<int>&, const std::vector<int>&, const std::vector<int>&, const std::vector<int>&, std::vector<int>&, std::vector<int>&, std::vector<int>&);
    template void Layers::transpose<float>(const std::vector<int>&, const std::vector<int>&, const std::vector<float>&, const std::vector<int>&, std::vector<int>&, std::vector<int>&, std::vector<float>&);
    template void Layers::transpose<double>(const std::vector<int>&, const std::vector<int>&, const std::vector<double>&, const std::vector<int>&, std::vector<int>&, std::vector<int>&, std::vector<double>&);
}