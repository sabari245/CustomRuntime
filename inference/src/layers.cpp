#include <vector>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <cmath>
#include <cassert>

#include "layers.h"

namespace layers {

    template <typename T>
    types::ArrayND<T> Layers::transpose(
        const types::ArrayND<T>& in,
        const std::vector<int>& permutation
    ) {
		return utility::transpose(in, permutation);
    }

    // Explicit template instantiations (if needed in a separate compilation unit)
    template types::ArrayND<int> Layers::transpose<int>(const types::ArrayND<int>&, const std::vector<int>&);
    template types::ArrayND<double> Layers::transpose<double>(const types::ArrayND<double>&, const std::vector<int>&);
    template types::ArrayND<uint8_t> Layers::transpose<uint8_t>(const types::ArrayND<uint8_t>&, const std::vector<int>&);

    template <typename T>
    types::ArrayND<T> Layers::convolutionOperation(
        const types::ArrayND<T>& data,
        const types::ArrayND<T>& kernel,
        const std::vector<int>& padding,  // [top, right, bottom, left]
        const std::vector<int>& stride
    ) {
        // Get dimensions
        const int channels = data.shape[0];     // Should be 3 for RGB
        const int height = data.shape[1];
        const int width = data.shape[2];
        const int kernelHeight = kernel.shape[1];
        const int kernelWidth = kernel.shape[2];

        //// Verify kernel shape (3, kernelHeight, kernelWidth)
        ////assert(kernel.shape[0] == 3 && "Kernel must have 3 channels");
        //assert(padding[0] == padding[1] && padding[1] == padding[2] && padding[2] == padding[3] &&
        //    "All padding values must be equal");

        // Add padding to the input data
        types::ArrayND<T> paddedData;
        if (padding[0] > 0) {  // Since all paddings are equal, just check one
            paddedData = utility::addPadding(data, padding);
        }
        else {
            paddedData = data;
        }

        // Calculate output dimensions
        const int outHeight = (paddedData.shape[1] - kernelHeight) / stride[0] + 1;
        const int outWidth = (paddedData.shape[2] - kernelWidth) / stride[1] + 1;

        // Create output array (single channel output)
        std::vector<int> outShape = { 1, outHeight, outWidth };
        types::ArrayND<T> output = utility::createZeros<T>(outShape);

        // Perform convolution
        for (int i = 0; i < outHeight; i++) {
            for (int j = 0; j < outWidth; j++) {
                T sum = 0;

                // Sum over all channels
                for (int c = 0; c < channels; c++) {
                    // Convolve at current position for this channel
                    for (int ki = 0; ki < kernelHeight; ki++) {
                        for (int kj = 0; kj < kernelWidth; kj++) {
                            int di = i * stride[0] + ki;
                            int dj = j * stride[1] + kj;

                            // Get values from padded input and kernel
                            std::vector<int> dataIdx = { c, di, dj };
                            std::vector<int> kernelIdx = { c, ki, kj };  // c is the channel filter

                            sum += utility::get(paddedData, dataIdx) *
                                utility::get(kernel, kernelIdx);
                        }
                    }
                }

                // Set output value
                std::vector<int> outIdx = { 0, i, j };  // Single channel output
                int flatIdx = 0;
                for (size_t k = 0; k < outIdx.size(); k++) {
                    flatIdx += outIdx[k] * output.stride[k];
                }
                output.data[flatIdx] = sum;
            }
        }

        return output;
    }

	template types::ArrayND<int> Layers::convolutionOperation<int>(
		const types::ArrayND<int>&,
		const types::ArrayND<int>&,
		const std::vector<int>&,
		const std::vector<int>&
	);

	template types::ArrayND<double> Layers::convolutionOperation<double>(
		const types::ArrayND<double>&,
		const types::ArrayND<double>&,
		const std::vector<int>&,
		const std::vector<int>&
	);

	template types::ArrayND<uint8_t> Layers::convolutionOperation<uint8_t>(
		const types::ArrayND<uint8_t>&,
		const types::ArrayND<uint8_t>&,
		const std::vector<int>&,
		const std::vector<int>&
	);

    template <typename T>
    types::ArrayND<T> Layers::conv2d(
        const types::ArrayND<T>& in,
        const types::ArrayND<T>& weights,
        const types::ArrayND<T>& bias,
        const std::vector<int>& padding,
        const std::vector<int>& stride,
        const std::vector<int>& kernel_shape
    ) {
        // Get dimensions
        const int batch_size = in.shape[0];     // Should be 1
        const int in_channels = in.shape[1];    // Should be 3 for RGB
        const int in_height = in.shape[2];
        const int in_width = in.shape[3];
    
        const int num_kernels = weights.shape[0];
    
        // Calculate output dimensions
        const int out_height = (in_height + padding[0] + padding[2] - kernel_shape[0]) / stride[0] + 1;
        const int out_width = (in_width + padding[1] + padding[3] - kernel_shape[1]) / stride[1] + 1;
    
        // Create output array
        std::vector<int> out_shape = {batch_size, num_kernels, out_height, out_width};
        types::ArrayND<T> output = utility::createZeros<T>(out_shape);
    
        // Process each input batch (though batch_size is always 1 in this case)
        for (int b = 0; b < batch_size; b++) {
            // Extract current batch
            std::vector<int> batch_start = {b, 0, 0, 0};
            std::vector<int> batch_end = {b + 1, in_channels, in_height, in_width};
            types::ArrayND<T> current_batch = utility::sliceND(in, batch_start, batch_end);
        
            // Process each kernel
            for (int k = 0; k < num_kernels; k++) {
                // Extract current kernel
                std::vector<int> kernel_start = {k, 0, 0, 0};
                std::vector<int> kernel_end = {k + 1, in_channels, kernel_shape[0], kernel_shape[1]};
                types::ArrayND<T> current_kernel = utility::sliceND(weights, kernel_start, kernel_end);
            
                // Reshape extracted data to match convolutionOperation expectations
                std::vector<int> new_batch_shape = {in_channels, in_height, in_width};
                std::vector<int> new_kernel_shape = {in_channels, kernel_shape[0], kernel_shape[1]};
                types::ArrayND<T> reshaped_batch = utility::reshape(current_batch, new_batch_shape);
                types::ArrayND<T> reshaped_kernel = utility::reshape(current_kernel, new_kernel_shape);
            
                // Perform convolution for current batch and kernel
                types::ArrayND<T> conv_result = convolutionOperation(
                    reshaped_batch,
                    reshaped_kernel,
                    padding,
                    stride
                );
            
                // Add bias
                conv_result = utility::add(conv_result, bias.data[k]);
            
                // Copy result to output array
                for (int i = 0; i < out_height; i++) {
                    for (int j = 0; j < out_width; j++) {
                        std::vector<int> out_idx = {b, k, i, j};
                        std::vector<int> conv_idx = {0, i, j}; // convolution result is single-channel
                    
                        int out_flat_idx = 0;
                        int conv_flat_idx = 0;
                    
                        for (size_t idx = 0; idx < out_idx.size(); idx++) {
                            out_flat_idx += out_idx[idx] * output.stride[idx];
                        }
                        for (size_t idx = 0; idx < conv_idx.size(); idx++) {
                            conv_flat_idx += conv_idx[idx] * conv_result.stride[idx];
                        }
                    
                        output.data[out_flat_idx] = conv_result.data[conv_flat_idx];
                    }
                }
            }
        }
    
        return output;
    }

	template types::ArrayND<int> Layers::conv2d<int>(
		const types::ArrayND<int>&,
		const types::ArrayND<int>&,
		const types::ArrayND<int>&,
		const std::vector<int>&,
		const std::vector<int>&,
		const std::vector<int>&
	);

	template types::ArrayND<double> Layers::conv2d<double>(
		const types::ArrayND<double>&,
		const types::ArrayND<double>&,
		const types::ArrayND<double>&,
		const std::vector<int>&,
		const std::vector<int>&,
		const std::vector<int>&
	);

	template types::ArrayND<uint8_t> Layers::conv2d<uint8_t>(
		const types::ArrayND<uint8_t>&,
		const types::ArrayND<uint8_t>&,
		const types::ArrayND<uint8_t>&,
		const std::vector<int>&,
		const std::vector<int>&,
		const std::vector<int>&
	);

	template <typename T>
	types::ArrayND<T> Layers::Relu(
		const types::ArrayND<T>& in
	) {
		types::ArrayND<T> out = in;
		for (int i = 0; i < in.data.size(); i++) {
			out.data[i] = std::max(in.data[i], 0);
		}
		return out;
	}

	template <typename T>
    types::ArrayND<T> Layers::Softmax(
        const types::ArrayND<T>& in
    ) {

        // e^(x_i) / sum(e^(x_i)) for all i

        types::ArrayND<T> exp_in = utility::createZeros<T>(in.shape);
        for (int i = 0; i < in.data.size(); i++) {
            exp_in.data[i] = std::exp(in.data[i]);
        }

        T exp_sum = std::accumulate(exp_in.data.begin(), exp_in.data.end(), 0.0);

        types::ArrayND<T> out = utility::createZeros<T>(in.shape);
        for (int i = 0; i < in.data.size(); i++) {
            out.data[i] = exp_in.data[i] / exp_sum;
        }
        return out;
    }

    template <typename T>
    types::ArrayND<T> Layers::maxPool(
        const types::ArrayND<T>& in,
        const std::vector<int>& kernel_shape, // [height, width]
        const std::vector<int>& stride        // [row, col]
    ) {
        const int batch_size = in.shape[0];  // Should be 1
        const int in_channels = in.shape[1]; // Should be 64 from the previous conv layer
        const int in_height = in.shape[2];
        const int in_width = in.shape[3];

        const int out_height = ((in_height - kernel_shape[0]) / stride[0]) + 1;
        const int out_width = ((in_width - kernel_shape[1]) / stride[1]) + 1;

        // Create output array
        std::vector<int> out_shape = { batch_size, in_channels, out_height, out_width };
        types::ArrayND<T> output = utility::createZeros<T>(out_shape);

        // Perform max pooling
        for (int b = 0; b < batch_size; b++) {
            for (int c = 0; c < in_channels; c++) {
                for (int oh = 0; oh < out_height; oh++) {
                    for (int ow = 0; ow < out_width; ow++) {
                        T max_value = std::numeric_limits<T>::lowest();

                        // Compute the region in the input corresponding to the current output element
                        int row_start = oh * stride[0];
                        int row_end = std::min(row_start + kernel_shape[0], in_height);
                        int col_start = ow * stride[1];
                        int col_end = std::min(col_start + kernel_shape[1], in_width);

                        for (int r = row_start; r < row_end; r++) {
                            for (int c_ = col_start; c_ < col_end; c_++) {
                                std::vector<int> index = { b, c, r, c_ };
                                T value = utility::get(in, index);
                                max_value = std::max(max_value, value);
                            }
                        }

                        // Set the max value in the output array
                        std::vector<int> output_index = { b, c, oh, ow };
                        output.data[utility::getOffset(output, output_index)] = max_value;
                    }
                }
            }
        }

        return output;
    }

	template types::ArrayND<int> Layers::maxPool<int>(const types::ArrayND<int>&, const std::vector<int>&, const std::vector<int>&);
	template types::ArrayND<double> Layers::maxPool<double>(const types::ArrayND<double>&, const std::vector<int>&, const std::vector<int>&);
	template types::ArrayND<uint8_t> Layers::maxPool<uint8_t>(const types::ArrayND<uint8_t>&, const std::vector<int>&, const std::vector<int>&);

    template <typename T>
    types::ArrayND<T> Layers::Reshape(
        const types::ArrayND<T>& in,
        const types::ArrayND<T>& shape
    ) {
        // Convert shape.data to integers
        std::vector<int> intShape;
        for (const auto& val : shape.data) {
            intShape.push_back(static_cast<int>(val));
        }

        // Call the reshape utility function with the converted shape
        return utility::reshape(in, intShape);
    }

	template <typename T>
	types::ArrayND<T> Layers::MatMul(
		const types::ArrayND<T>& in1,
		const types::ArrayND<T>& in2
	) {
		// Check if the matrices can be multiplied
		if (in1.shape[1] != in2.shape[0]) {
			throw std::invalid_argument("Matrix dimensions do not match for multiplication");
		}
		// Create the output matrix
		std::vector<int> outShape = { in1.shape[0], in2.shape[1] };
		types::ArrayND<T> out = utility::createZeros<T>(outShape);
		// Perform matrix multiplication
		for (int i = 0; i < in1.shape[0]; i++) {
			for (int j = 0; j < in2.shape[1]; j++) {
				T sum = 0;
				for (int k = 0; k < in1.shape[1]; k++) {
					sum += in1.data[i * in1.shape[1] + k] * in2.data[k * in2.shape[1] + j];
				}
				out.data[i * out.shape[1] + j] = sum;
			}
		}
		return out;
	}

	template types::ArrayND<int> Layers::MatMul<int>(const types::ArrayND<int>&, const types::ArrayND<int>&);
	template types::ArrayND<double> Layers::MatMul<double>(const types::ArrayND<double>&, const types::ArrayND<double>&);
	template types::ArrayND<uint8_t> Layers::MatMul<uint8_t>(const types::ArrayND<uint8_t>&, const types::ArrayND<uint8_t>&);
    
    template <typename T>
    types::ArrayND<T> Layers::Add(
        const types::ArrayND<T>& in1,
        const types::ArrayND<T>& in2
    ) {
        if (in1.shape == in2.shape) {
            return utility::add(in1, in2);
        }

        if (in1.shape.size() != 4 || in2.shape.size() != 4) {
            throw std::runtime_error("Both inputs must be 4D arrays");
        }

        // Create output array with same shape as in1
        types::ArrayND<T> output;
        output.shape = in1.shape;
        output.stride = utility::getStrideFromShape(output.shape);
        output.data.resize(in1.data.size());

        // Calculate strides for faster access
        const auto& stride1 = in1.stride;
        const auto& stride2 = in2.stride;

        // Handle broadcasting
        for (int n = 0; n < in1.shape[0]; ++n) {
            for (int c = 0; c < in1.shape[1]; ++c) {
                for (int h = 0; h < in1.shape[2]; ++h) {
                    for (int w = 0; w < in1.shape[3]; ++w) {
                        // Calculate indices for both arrays
                        std::vector<int> idx1 = { n, c, h, w };
                        std::vector<int> idx2 = { n, c,
                            h % in2.shape[2],
                            w % in2.shape[3]
                        };

                        int offset1 = utility::getOffset(in1, idx1);
                        int offset2 = utility::getOffset(in2, idx2);
                        int offsetOut = n * stride1[0] + c * stride1[1] +
                            h * stride1[2] + w * stride1[3];

                        output.data[offsetOut] = in1.data[offset1] + in2.data[offset2];
                    }
                }
            }
        }

        return output;
    }

	template types::ArrayND<int> Layers::Add<int>(const types::ArrayND<int>&, const types::ArrayND<int>&);
	template types::ArrayND<double> Layers::Add<double>(const types::ArrayND<double>&, const types::ArrayND<double>&);
	template types::ArrayND<uint8_t> Layers::Add<uint8_t>(const types::ArrayND<uint8_t>&, const types::ArrayND<uint8_t>&);

    template <typename T>
    types::ArrayND<T> Layers::Mul(
        const types::ArrayND<T>& in1,
        const types::ArrayND<T>& in2
    ) {
        // First check if shapes are identical
        if (in1.shape == in2.shape) {
            // Use the regular multiply function for same shape arrays
            return utility::mul(in1, in2);
        }

        // Handle broadcasting case
        // Verify input dimensions are 4D
        if (in1.shape.size() != 4 || in2.shape.size() != 4) {
            throw std::runtime_error("Both inputs must be 4D arrays");
        }

        // Create output array with same shape as in1
        types::ArrayND<T> output;
        output.shape = in1.shape;
        output.stride = utility::getStrideFromShape(output.shape);
        output.data.resize(in1.data.size());

        // Calculate strides for faster access
        const auto& stride1 = in1.stride;
        const auto& stride2 = in2.stride;

        // Handle broadcasting
        for (int n = 0; n < in1.shape[0]; ++n) {           // Batch
            for (int c = 0; c < in1.shape[1]; ++c) {       // Channels
                for (int h = 0; h < in1.shape[2]; ++h) {   // Height
                    for (int w = 0; w < in1.shape[3]; ++w) { // Width
                        // Calculate indices for both arrays
                        std::vector<int> idx1 = { n, c, h, w };
                        std::vector<int> idx2 = { n, c,
                            // Use 0 for height and width if in2 has shape 1 in those dimensions
                            h % in2.shape[2],
                            w % in2.shape[3]
                        };

                        // Calculate offsets
                        int offset1 = utility::getOffset(in1, idx1);
                        int offset2 = utility::getOffset(in2, idx2);
                        int offsetOut = n * stride1[0] + c * stride1[1] +
                            h * stride1[2] + w * stride1[3];

                        // Perform multiplication
                        output.data[offsetOut] = in1.data[offset1] * in2.data[offset2];
                    }
                }
            }
        }

        return output;
    }

	template types::ArrayND<int> Layers::Mul<int>(const types::ArrayND<int>&, const types::ArrayND<int>&);
	template types::ArrayND<double> Layers::Mul<double>(const types::ArrayND<double>&, const types::ArrayND<double>&);
	template types::ArrayND<uint8_t> Layers::Mul<uint8_t>(const types::ArrayND<uint8_t>&, const types::ArrayND<uint8_t>&);

}