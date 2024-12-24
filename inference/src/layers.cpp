#include <vector>
#include <algorithm>
#include <numeric>
#include <stdexcept>

#include "layers.h"

namespace layers {

    template <typename T>
    void Layers::transpose(
        const ArrayND<T>& in,
        const std::vector<int>& permutation,
        ArrayND<T>& out
    ) {
        // Validate input
        if (in.shape.size() != permutation.size()) {
            throw std::invalid_argument("Permutation size must match matrix dimensions");
        }

        // Rearrange shape according to permutation
        out.shape.resize(in.shape.size());
        for (size_t i = 0; i < permutation.size(); i++) {
            out.shape[i] = in.shape[permutation[i]];
        }

        // Calculate new strides
        out.stride.resize(in.shape.size());
        int current_stride = 1;
        for (int i = out.shape.size() - 1; i >= 0; i--) {
            out.stride[i] = current_stride;
            current_stride *= out.shape[i];
        }

        // Initialize output data
        int total_size = std::accumulate(in.shape.begin(), in.shape.end(), 1, std::multiplies<int>());
        out.data.resize(total_size);

        // Create index arrays for iteration
        std::vector<int> current_idx(in.shape.size(), 0);

        // Iterate through all elements
        for (int flat_idx = 0; flat_idx < total_size; flat_idx++) {
            // Calculate source index
            int src_idx = 0;
            for (size_t dim = 0; dim < in.shape.size(); dim++) {
                src_idx += current_idx[dim] * in.stride[dim];
            }

            // Calculate destination index
            int dst_idx = 0;
            for (size_t dim = 0; dim < out.shape.size(); dim++) {
                dst_idx += current_idx[permutation[dim]] * out.stride[dim];
            }

            // Copy data
            out.data[dst_idx] = in.data[src_idx];

            // Update indices
            for (int dim = in.shape.size() - 1; dim >= 0; dim--) {
                current_idx[dim]++;
                if (current_idx[dim] < in.shape[dim]) {
                    break;
                }
                current_idx[dim] = 0;
            }
        }
    }

    // Explicit template instantiations (if needed in a separate compilation unit)
    template void Layers::transpose<int>(const ArrayND<int>&, const std::vector<int>&, ArrayND<int>&);
    template void Layers::transpose<float>(const ArrayND<float>&, const std::vector<int>&, ArrayND<float>&);
    template void Layers::transpose<double>(const ArrayND<double>&, const std::vector<int>&, ArrayND<double>&);
	template void Layers::transpose<uint8_t>(const ArrayND<uint8_t>&, const std::vector<int>&, ArrayND<uint8_t>&);
}