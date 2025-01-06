#pragma once

#include <vector>
#include "types.h"
#include "functions.h"

namespace layers
{
	class Layers
	{
	private:
		// Helper function for 3x3 kernel convolution
		template <typename T>
		static inline T convolve3x3(
			const std::vector<T> &input_data, const std::vector<T> &weight_data,
			const size_t input_offset, const size_t weight_offset,
			const int padded_stride_c, const int padded_stride_h, const int padded_stride_w,
			const int weight_stride_c, const int weight_stride_h, const int weight_stride_w,
			const int in_channels)
		{

			T sum = 0;
			for (int c = 0; c < in_channels; c++)
			{
				const size_t in_c_offset = input_offset + c * padded_stride_c;
				const size_t w_c_offset = weight_offset + c * weight_stride_c;

				// Unrolled 3x3 kernel multiplication
				sum += input_data[in_c_offset + 0 * padded_stride_h + 0 * padded_stride_w] *
					   weight_data[w_c_offset + 0 * weight_stride_h + 0 * weight_stride_w];
				sum += input_data[in_c_offset + 0 * padded_stride_h + 1 * padded_stride_w] *
					   weight_data[w_c_offset + 0 * weight_stride_h + 1 * weight_stride_w];
				sum += input_data[in_c_offset + 0 * padded_stride_h + 2 * padded_stride_w] *
					   weight_data[w_c_offset + 0 * weight_stride_h + 2 * weight_stride_w];

				sum += input_data[in_c_offset + 1 * padded_stride_h + 0 * padded_stride_w] *
					   weight_data[w_c_offset + 1 * weight_stride_h + 0 * weight_stride_w];
				sum += input_data[in_c_offset + 1 * padded_stride_h + 1 * padded_stride_w] *
					   weight_data[w_c_offset + 1 * weight_stride_h + 1 * weight_stride_w];
				sum += input_data[in_c_offset + 1 * padded_stride_h + 2 * padded_stride_w] *
					   weight_data[w_c_offset + 1 * weight_stride_h + 2 * weight_stride_w];

				sum += input_data[in_c_offset + 2 * padded_stride_h + 0 * padded_stride_w] *
					   weight_data[w_c_offset + 2 * weight_stride_h + 0 * weight_stride_w];
				sum += input_data[in_c_offset + 2 * padded_stride_h + 1 * padded_stride_w] *
					   weight_data[w_c_offset + 2 * weight_stride_h + 1 * weight_stride_w];
				sum += input_data[in_c_offset + 2 * padded_stride_h + 2 * padded_stride_w] *
					   weight_data[w_c_offset + 2 * weight_stride_h + 2 * weight_stride_w];
			}
			return sum;
		}

		// Helper function for 5x5 kernel convolution
		template <typename T>
		static inline T convolve5x5(
			const std::vector<T> &input_data, const std::vector<T> &weight_data,
			const size_t input_offset, const size_t weight_offset,
			const int padded_stride_c, const int padded_stride_h, const int padded_stride_w,
			const int weight_stride_c, const int weight_stride_h, const int weight_stride_w,
			const int in_channels)
		{

			T sum = 0;
			for (int c = 0; c < in_channels; c++)
			{
				const size_t in_c_offset = input_offset + c * padded_stride_c;
				const size_t w_c_offset = weight_offset + c * weight_stride_c;

				// Unrolled 5x5 kernel multiplication
				for (int i = 0; i < 5; i++)
				{
					for (int j = 0; j < 5; j++)
					{
						sum += input_data[in_c_offset + i * padded_stride_h + j * padded_stride_w] *
							   weight_data[w_c_offset + i * weight_stride_h + j * weight_stride_w];
					}
				}
			}
			return sum;
		}

	public:
		template <typename T>
		types::ArrayND<T> transpose(
			const types::ArrayND<T> &in,
			const std::vector<int> &permutation);

		template <typename T>
		types::ArrayND<T> conv2d(
			const types::ArrayND<T> &in,
			const types::ArrayND<T> &weights,
			const types::ArrayND<T> &bias,
			const std::vector<int> &padding,
			const std::vector<int> &stride,
			const std::vector<int> &kernel_shape);

		template <typename T>
		types::ArrayND<T> Relu(
			const types::ArrayND<T> &in);

		template <typename T>
		void Relu_Inplace(
			types::ArrayND<T> &in);

		template <typename T>
		types::ArrayND<T> maxPool(
			const types::ArrayND<T> &in,
			const std::vector<int> &kernel_shape,
			const std::vector<int> &stride);

		template <typename T>
		types::ArrayND<T> Reshape(
			const types::ArrayND<T> &in,
			const types::ArrayND<T> &shape);

		template <typename T>
		types::ArrayND<T> MatMul(
			const types::ArrayND<T> &in1,
			const types::ArrayND<T> &in2);

		template <typename T>
		types::ArrayND<T> Add(
			const types::ArrayND<T> &in1,
			const types::ArrayND<T> &in2);

		template <typename T>
		void Add_Inplace(
			types::ArrayND<T> &in1,
			const types::ArrayND<T> &in2);

		template <typename T>
		types::ArrayND<T> Mul(
			const types::ArrayND<T> &in1,
			const types::ArrayND<T> &in2);

		template <typename T>
		void Mul_Inplace(
			types::ArrayND<T> &in1,
			const types::ArrayND<T> &in2);

		template <typename T>
		void Mul_And_Add(
			types::ArrayND<T> &in1,
			const types::ArrayND<T> &in2,
			const types::ArrayND<T> &in3);

		template <typename T>
		types::ArrayND<T> Softmax(
			const types::ArrayND<T> &in);
	};
}