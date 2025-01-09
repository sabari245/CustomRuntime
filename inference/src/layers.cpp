#include <vector>
#include <numeric>
#include <stdexcept>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <iostream>

#include "fftw3.h"
#include "layers.h"

namespace layers
{

    template <typename T>
    void Layers::fft2d(
        const types::ArrayND<T> &in,
        types::ArrayND<std::complex<double>> &out)
    {
        if (in.shape.size() != 2)
        {
            throw std::invalid_argument("Input array must be 2D");
        }

        fftw_complex *in_fft = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * in.shape[0] * in.shape[1]);
        fftw_complex *out_fft = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * in.shape[0] * in.shape[1]);

        fftw_plan plan = fftw_plan_dft_2d(in.shape[0], in.shape[1], in_fft, out_fft, FFTW_FORWARD, FFTW_ESTIMATE);

        for (int i = 0; i < in.shape[0] * in.shape[1]; i++)
        {
            in_fft[i][0] = in.data[i];
            in_fft[i][1] = 0;
        }

        fftw_execute(plan);

        for (int i = 0; i < in.shape[0] * in.shape[1]; i++)
        {
            out.data[i] = std::complex<double>(out_fft[i][0], out_fft[i][1]);
        }

        fftw_destroy_plan(plan);
        fftw_free(in_fft);
        fftw_free(out_fft);
    }

    template void Layers::fft2d(const types::ArrayND<double> &, types::ArrayND<std::complex<double>> &);
    template void Layers::fft2d(const types::ArrayND<int> &, types::ArrayND<std::complex<double>> &);

    template <typename T>
    void Layers::ifft2d(
        const types::ArrayND<std::complex<double>> &in,
        types::ArrayND<T> &out)
    {
        if (in.shape.size() != 2)
        {
            throw std::invalid_argument("Input array must be 2D");
        }

        fftw_complex *in_fft = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * in.shape[0] * in.shape[1]);
        fftw_complex *out_fft = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * in.shape[0] * in.shape[1]);
        fftw_plan plan = fftw_plan_dft_2d(in.shape[0], in.shape[1], in_fft, out_fft, FFTW_BACKWARD, FFTW_ESTIMATE);

        for (int i = 0; i < in.shape[0] * in.shape[1]; i++)
        {
            in_fft[i][0] = in.data[i].real();
            in_fft[i][1] = in.data[i].imag();
        }

        fftw_execute(plan);

        for (int i = 0; i < in.shape[0] * in.shape[1]; i++)
        {
            out.data[i] = out_fft[i][0] / (in.shape[0] * in.shape[1]);
        }

        fftw_destroy_plan(plan);
        fftw_free(in_fft);
        fftw_free(out_fft);
    }

    template void Layers::ifft2d(const types::ArrayND<std::complex<double>> &, types::ArrayND<double> &);
    template void Layers::ifft2d(const types::ArrayND<std::complex<double>> &, types::ArrayND<int> &);

    template <typename T>
    types::ArrayND<T> Layers::convolveFFT(
        const types::ArrayND<T> &image,
        const types::ArrayND<T> &filter)
    {
        if (image.shape.size() != 2 || filter.shape.size() != 2)
        {
            throw std::invalid_argument("Input arrays must be 2D");
        }
        int imageHeight = image.shape[0];
        int imageWidth = image.shape[1];
        int filterHeight = filter.shape[0];
        int filterWidth = filter.shape[1];

        // padding the kernel
        types::ArrayND<T> paddedFilter = utility::createZeros<T>({imageHeight, imageWidth});
        for (int i = 0; i < filterHeight; i++)
        {
            for (int j = 0; j < filterWidth; j++)
            {
                paddedFilter.data[i * imageWidth + j] = filter.data[i * filterWidth + j];
            }
        }

        // FFT the image and the kernel
        types::ArrayND<std::complex<double>> fftImage, fftFilter;
        fftImage.data.resize(image.data.size());
        fftFilter.data.resize(image.data.size());
        fftImage.shape = image.shape;
        fftFilter.shape = image.shape;
        fftImage.stride = image.stride;
        fftFilter.stride = image.stride; // All the size, shape and strides are the same
        fft2d(image, fftImage);
        fft2d(paddedFilter, fftFilter);

        // Multiply the FFTs
        types::ArrayND<std::complex<double>> fftResult;
        fftResult.shape = image.shape;
        fftResult.stride = image.stride;
        fftResult.data.resize(image.data.size());

        for (int i = 0; i < image.data.size(); i++)
        {
            fftResult.data[i] = fftImage.data[i] * fftFilter.data[i];
        }

        // IFFT the result
        types::ArrayND<T> result;
        result.shape = image.shape;
        result.stride = image.stride;
        result.data.resize(image.data.size());

        ifft2d(fftResult, result);

        int validRows = imageHeight - filterHeight + 1;
        int validCols = imageWidth - filterWidth + 1;
        types::ArrayND<T> validResult = utility::createZeros<T>({validRows, validCols});

        // the result is stored in the bottom right corner of the result array (due to the circular convolution)
        int incrementRows = imageHeight - validRows;
        int incrementCols = imageWidth - validCols;

        for (int i = 0; i < validRows; i++)
        {
            for (int j = 0; j < validCols; j++)
            {
                validResult.data[i * validCols + j] = result.data[(i + incrementRows) * imageWidth + j + incrementCols];
            }
        }

        return validResult;
    }

    template types::ArrayND<double> Layers::convolveFFT<double>(
        const types::ArrayND<double> &,
        const types::ArrayND<double> &);

    template types::ArrayND<int> Layers::convolveFFT<int>(
        const types::ArrayND<int> &,
        const types::ArrayND<int> &);

    template <typename T>
    types::ArrayND<T> Layers::transpose(
        const types::ArrayND<T> &in,
        const std::vector<int> &permutation)
    {
        return utility::transpose(in, permutation);
    }

    // Explicit template instantiations (if needed in a separate compilation unit)
    template types::ArrayND<int> Layers::transpose<int>(const types::ArrayND<int> &, const std::vector<int> &);
    template types::ArrayND<double> Layers::transpose<double>(const types::ArrayND<double> &, const std::vector<int> &);
    template types::ArrayND<uint8_t> Layers::transpose<uint8_t>(const types::ArrayND<uint8_t> &, const std::vector<int> &);

    template <typename T>
    types::ArrayND<T> Layers::conv2d(
        const types::ArrayND<T> &in,
        const types::ArrayND<T> &weights,
        const types::ArrayND<T> &bias,
        const std::vector<int> &padding, // [top, right, bottom, left]
        const std::vector<int> &stride,
        const std::vector<int> &kernel_shape)
    {
        // Get dimensions
        const int batch_size = in.shape[0];
        const int in_channels = in.shape[1];
        const int in_height = in.shape[2];
        const int in_width = in.shape[3];

        const int num_kernels = weights.shape[0];

        types::ArrayND<T> paddedData;
        if (padding[0] > 0 || padding[1] > 0 || padding[2] > 0 || padding[3] > 0)
        {
            paddedData = utility::addPadding(in, padding);
        }
        else
        {
            paddedData = in;
        }

        const int out_height = (in_height + padding[0] + padding[2] - kernel_shape[0]) / stride[0] + 1;
        const int out_width = (in_width + padding[1] + padding[3] - kernel_shape[1]) / stride[1] + 1;

        std::vector<int> out_shape = {batch_size, num_kernels, out_height, out_width};
        types::ArrayND<T> output = utility::createZeros<T>(out_shape);

        const int padded_stride_b = paddedData.stride[0];
        const int padded_stride_c = paddedData.stride[1];
        const int padded_stride_h = paddedData.stride[2];
        const int padded_stride_w = paddedData.stride[3];

        const int weight_stride_k = weights.stride[0];
        const int weight_stride_c = weights.stride[1];
        const int weight_stride_h = weights.stride[2];
        const int weight_stride_w = weights.stride[3];

        const int out_stride_b = output.stride[0];
        const int out_stride_k = output.stride[1];
        const int out_stride_h = output.stride[2];
        const int out_stride_w = output.stride[3];

        // Process each input batch
        for (int b = 0; b < batch_size; b++)
        {
            // Process each kernel
            for (int k = 0; k < num_kernels; k++)
            {
                // Process each output position
                for (int oh = 0; oh < out_height; oh++)
                {
                    for (int ow = 0; ow < out_width; ow++)
                    {
                        T sum = 0;

                        // Convolve at current position
                        for (int c = 0; c < in_channels; c++)
                        {
                            for (int kh = 0; kh < kernel_shape[0]; kh++)
                            {
                                for (int kw = 0; kw < kernel_shape[1]; kw++)
                                {
                                    int ih = oh * stride[0] + kh;
                                    int iw = ow * stride[1] + kw;

                                    sum += paddedData.data[b * padded_stride_b + c * padded_stride_c + ih * padded_stride_h + iw * padded_stride_w] *
                                           weights.data[k * weight_stride_k + c * weight_stride_c + kh * weight_stride_h + kw * weight_stride_w];
                                }
                            }
                        }

                        // Add bias and store result
                        int out_idx = b * out_stride_b + k * out_stride_k + oh * out_stride_h + ow * out_stride_w;
                        output.data[out_idx] = sum + bias.data[k];
                    }
                }
            }
        }

        return output;
    }

    template types::ArrayND<int> Layers::conv2d<int>(
        const types::ArrayND<int> &,
        const types::ArrayND<int> &,
        const types::ArrayND<int> &,
        const std::vector<int> &,
        const std::vector<int> &,
        const std::vector<int> &);

    template types::ArrayND<double> Layers::conv2d<double>(
        const types::ArrayND<double> &,
        const types::ArrayND<double> &,
        const types::ArrayND<double> &,
        const std::vector<int> &,
        const std::vector<int> &,
        const std::vector<int> &);

    template types::ArrayND<uint8_t> Layers::conv2d<uint8_t>(
        const types::ArrayND<uint8_t> &,
        const types::ArrayND<uint8_t> &,
        const types::ArrayND<uint8_t> &,
        const std::vector<int> &,
        const std::vector<int> &,
        const std::vector<int> &);

    template <typename T>
    types::ArrayND<T> Layers::conv2dFFT(
        const types::ArrayND<T> &in,
        const types::ArrayND<T> &weights,
        const types::ArrayND<T> &bias,
        const std::vector<int> &padding, // [top, right, bottom, left]
        const std::vector<int> &stride,
        const std::vector<int> &kernel_shape)
    {
        // Get dimensions
        const int batch_size = in.shape[0];
        const int in_channels = in.shape[1];
        const int in_height = in.shape[2];
        const int in_width = in.shape[3];

        const int num_kernels = weights.shape[0];

        types::ArrayND<T> paddedData;
        if (padding[0] > 0 || padding[1] > 0 || padding[2] > 0 || padding[3] > 0)
        {
            paddedData = utility::addPadding(in, padding);
        }
        else
        {
            paddedData = in;
        }

        const int out_height = (in_height + padding[0] + padding[2] - kernel_shape[0]) / stride[0] + 1;
        const int out_width = (in_width + padding[1] + padding[3] - kernel_shape[1]) / stride[1] + 1;

        std::vector<int> out_shape = {batch_size, num_kernels, out_height, out_width};
        types::ArrayND<T> output = utility::createZeros<T>(out_shape);

        const int padded_stride_b = paddedData.stride[0];
        const int padded_stride_c = paddedData.stride[1];
        const int padded_stride_h = paddedData.stride[2];
        const int padded_stride_w = paddedData.stride[3];

        const int weight_stride_k = weights.stride[0];
        const int weight_stride_c = weights.stride[1];
        const int weight_stride_h = weights.stride[2];
        const int weight_stride_w = weights.stride[3];

        const int out_stride_b = output.stride[0];
        const int out_stride_k = output.stride[1];
        const int out_stride_h = output.stride[2];
        const int out_stride_w = output.stride[3];

        // Process each input batch
        for (int b = 0; b < batch_size; b++)
        {
            // Process each kernel
            for (int k = 0; k < num_kernels; k++)
            {
                // Zero the output positions before accumulating
                for (int i = 0; i < out_height; i++)
                {
                    for (int j = 0; j < out_width; j++)
                    {
                        output.data[b * out_stride_b + k * out_stride_k + i * out_stride_h + j * out_stride_w] = 0;
                    }
                }

                // process each channel
                for (int c = 0; c < in_channels; c++)
                {
                    auto input_channel = utility::sliceND(paddedData, {b, c, 0, 0}, {b + 1, c + 1, paddedData.shape[2], paddedData.shape[3]});
                    utility::reshape_inplace(input_channel, {paddedData.shape[2], paddedData.shape[3]});

                    auto filter = utility::sliceND(weights, {k, c, 0, 0}, {k + 1, c + 1, kernel_shape[0], kernel_shape[1]});
                    utility::reshape_inplace(filter, {kernel_shape[0], kernel_shape[1]});

                    auto conv_result = convolveFFT(input_channel, filter);

                    // Apply stride when accumulating results
                    for (int i = 0; i < out_height; i++)
                    {
                        for (int j = 0; j < out_width; j++)
                        {
                            int src_i = i * stride[0];
                            int src_j = j * stride[1];
                            output.data[b * out_stride_b + k * out_stride_k + i * out_stride_h + j * out_stride_w] += conv_result.data[src_i * conv_result.shape[1] + src_j];
                        }
                    }
                }

                // Add bias after accumulating all channels
                for (int i = 0; i < out_height; i++)
                {
                    for (int j = 0; j < out_width; j++)
                    {
                        output.data[b * out_stride_b + k * out_stride_k + i * out_stride_h + j * out_stride_w] += bias.data[k];
                    }
                }
            }
        }

        return output;
    }

    template types::ArrayND<int> Layers::conv2dFFT<int>(
        const types::ArrayND<int> &,
        const types::ArrayND<int> &,
        const types::ArrayND<int> &,
        const std::vector<int> &,
        const std::vector<int> &,
        const std::vector<int> &);

    template types::ArrayND<double> Layers::conv2dFFT<double>(
        const types::ArrayND<double> &,
        const types::ArrayND<double> &,
        const types::ArrayND<double> &,
        const std::vector<int> &,
        const std::vector<int> &,
        const std::vector<int> &);

    template <typename T>
    types::ArrayND<T> Layers::Relu(
        const types::ArrayND<T> &in)
    {
        types::ArrayND<T> out = in;
        for (int i = 0; i < in.data.size(); i++)
        {
            if (in.data[i] < 0)
            {
                out.data[i] = 0;
            }
        }
        return out;
    }

    template types::ArrayND<int> Layers::Relu<int>(const types::ArrayND<int> &);
    template types::ArrayND<double> Layers::Relu<double>(const types::ArrayND<double> &);
    template types::ArrayND<uint8_t> Layers::Relu<uint8_t>(const types::ArrayND<uint8_t> &);

    template <typename T>
    void Layers::Relu_Inplace(
        types::ArrayND<T> &in)
    {
        std::transform(in.data.begin(), in.data.end(), in.data.begin(),
                       [](T x)
                       { return std::max(x, static_cast<T>(0)); });
    }

    template void Layers::Relu_Inplace<int>(types::ArrayND<int> &);
    template void Layers::Relu_Inplace<double>(types::ArrayND<double> &);
    template void Layers::Relu_Inplace<uint8_t>(types::ArrayND<uint8_t> &);

    template <typename T>
    types::ArrayND<T> Layers::Softmax(
        const types::ArrayND<T> &in)
    {

        // e^(x_i) / sum(e^(x_i)) for all i

        types::ArrayND<T> exp_in = utility::createZeros<T>(in.shape);
        for (int i = 0; i < in.data.size(); i++)
        {
            exp_in.data[i] = std::exp(in.data[i]);
        }

        T exp_sum = std::accumulate(exp_in.data.begin(), exp_in.data.end(), 0.0);

        types::ArrayND<T> out = utility::createZeros<T>(in.shape);
        for (int i = 0; i < in.data.size(); i++)
        {
            out.data[i] = exp_in.data[i] / exp_sum;
        }
        return out;
    }

    // template types::ArrayND<int> Layers::Softmax<int>(const types::ArrayND<int>&);
    template types::ArrayND<double> Layers::Softmax<double>(const types::ArrayND<double> &);
    // template types::ArrayND<uint8_t> Layers::Softmax<uint8_t>(const types::ArrayND<uint8_t>&);

    template <typename T>
    types::ArrayND<T> Layers::maxPool(
        const types::ArrayND<T> &in,
        const std::vector<int> &kernel_shape, // [height, width]
        const std::vector<int> &stride        // [row, col]
    )
    {
        const int batch_size = in.shape[0];  // Should be 1
        const int in_channels = in.shape[1]; // Should be 64 from the previous conv layer
        const int in_height = in.shape[2];
        const int in_width = in.shape[3];

        const int out_height = ((in_height - kernel_shape[0]) / stride[0]) + 1;
        const int out_width = ((in_width - kernel_shape[1]) / stride[1]) + 1;

        // Create output array
        std::vector<int> out_shape = {batch_size, in_channels, out_height, out_width};
        types::ArrayND<T> output = utility::createZeros<T>(out_shape);

        const int intputStride0 = in.stride[0];
        const int intputStride1 = in.stride[1];
        const int intputStride2 = in.stride[2];
        const int intputStride3 = in.stride[3];

        const int outputStride0 = output.stride[0];
        const int outputStride1 = output.stride[1];
        const int outputStride2 = output.stride[2];
        const int outputStride3 = output.stride[3];

        // Perform max pooling
        for (int b = 0; b < batch_size; b++)
        {
            for (int c = 0; c < in_channels; c++)
            {
                for (int oh = 0; oh < out_height; oh++)
                {
                    for (int ow = 0; ow < out_width; ow++)
                    {
                        T max_value = std::numeric_limits<T>::lowest();

                        // Compute the region in the input corresponding to the current output element
                        int row_start = oh * stride[0];
                        int row_end = std::min(row_start + kernel_shape[0], in_height);
                        int col_start = ow * stride[1];
                        int col_end = std::min(col_start + kernel_shape[1], in_width);

                        for (int r = row_start; r < row_end; r++)
                        {
                            for (int c_ = col_start; c_ < col_end; c_++)
                            {
                                std::vector<int> index = {b, c, r, c_};
                                max_value = std::max(max_value, in.data[b * intputStride0 + c * intputStride1 + r * intputStride2 + c_ * intputStride3]);
                            }
                        }

                        output.data[b * outputStride0 + c * outputStride1 + oh * outputStride2 + ow * outputStride3] = max_value;
                    }
                }
            }
        }

        return output;
    }

    template types::ArrayND<int> Layers::maxPool<int>(const types::ArrayND<int> &, const std::vector<int> &, const std::vector<int> &);
    template types::ArrayND<double> Layers::maxPool<double>(const types::ArrayND<double> &, const std::vector<int> &, const std::vector<int> &);
    template types::ArrayND<uint8_t> Layers::maxPool<uint8_t>(const types::ArrayND<uint8_t> &, const std::vector<int> &, const std::vector<int> &);

    template <typename T>
    types::ArrayND<T> Layers::Reshape(
        const types::ArrayND<T> &in,
        const types::ArrayND<T> &shape)
    {
        // Convert shape.data to integers
        std::vector<int> intShape;
        for (const auto &val : shape.data)
        {
            intShape.push_back(static_cast<int>(val));
        }

        // Call the reshape utility function with the converted shape
        return utility::reshape(in, intShape);
    }

    template types::ArrayND<int> Layers::Reshape<int>(const types::ArrayND<int> &, const types::ArrayND<int> &);
    template types::ArrayND<double> Layers::Reshape<double>(const types::ArrayND<double> &, const types::ArrayND<double> &);
    template types::ArrayND<uint8_t> Layers::Reshape<uint8_t>(const types::ArrayND<uint8_t> &, const types::ArrayND<uint8_t> &);

    template <typename T>
    types::ArrayND<T> Layers::MatMul(
        const types::ArrayND<T> &in1,
        const types::ArrayND<T> &in2)
    {
        // Check if the matrices can be multiplied
        if (in1.shape[1] != in2.shape[0])
        {
            throw std::invalid_argument("Matrix dimensions do not match for multiplication");
        }
        // Create the output matrix
        // std::vector<int> outShape = ;
        types::ArrayND<T> out = utility::createZeros<T>({in1.shape[0], in2.shape[1]});
        // Perform matrix multiplication
        for (int i = 0; i < in1.shape[0]; i++)
        {
            for (int j = 0; j < in2.shape[1]; j++)
            {
                for (int k = 0; k < in1.shape[1]; k++)
                {
                    out.data[i * out.shape[1] + j] += in1.data[i * in1.shape[1] + k] * in2.data[k * in2.shape[1] + j];
                }
            }
        }
        return out;
    }

    template types::ArrayND<int> Layers::MatMul<int>(const types::ArrayND<int> &, const types::ArrayND<int> &);
    template types::ArrayND<double> Layers::MatMul<double>(const types::ArrayND<double> &, const types::ArrayND<double> &);
    template types::ArrayND<uint8_t> Layers::MatMul<uint8_t>(const types::ArrayND<uint8_t> &, const types::ArrayND<uint8_t> &);

    template <typename T>
    types::ArrayND<T> Layers::Add(
        const types::ArrayND<T> &in1, // {1, channels, height, width }
        const types::ArrayND<T> &in2  // {1, bias_per_feature, 1, 1}
    )
    {
        // just add them if they are of the same shape - bias addition case
        if (in1.shape == in2.shape)
        {
            return utility::add(in1, in2);
        }

        // if they are not of the same shape, then we need to broadcast the second input - batch normalization case
        if (in1.shape.size() != 4 || in2.shape.size() != 4)
        {
            throw std::runtime_error("Both inputs must be 4D arrays");
        }

        types::ArrayND<T> output = utility::createZeros<T>(in1.shape);

        // Calculate strides for faster access
        const int in1Stride0 = in1.stride[0];
        const int in1Stride1 = in1.stride[1];
        const int in1Stride2 = in1.stride[2];
        const int in1Stride3 = in1.stride[3];

        // Handle broadcasting
        for (int n = 0; n < in1.shape[0]; ++n)
        {
            for (int c = 0; c < in1.shape[1]; ++c)
            { // channels - this contains the addition value on the second input
                for (int h = 0; h < in1.shape[2]; ++h)
                {
                    for (int w = 0; w < in1.shape[3]; ++w)
                    {
                        int in1Idx = n * in1Stride0 + c * in1Stride1 + h * in1Stride2 + w * in1Stride3;
                        output.data[in1Idx] = in1.data[in1Idx] + in2.data[c];
                    }
                }
            }
        }

        return output;
    }

    template types::ArrayND<int> Layers::Add<int>(const types::ArrayND<int> &, const types::ArrayND<int> &);
    template types::ArrayND<double> Layers::Add<double>(const types::ArrayND<double> &, const types::ArrayND<double> &);
    template types::ArrayND<uint8_t> Layers::Add<uint8_t>(const types::ArrayND<uint8_t> &, const types::ArrayND<uint8_t> &);

    template <typename T>
    void Layers::Add_Inplace(types::ArrayND<T> &in1, const types::ArrayND<T> &in2)
    {
        if (in1.shape == in2.shape)
        {
            return utility::add_inplace(in1, in2);
        }

        // if they are not of the same shape, then we need to broadcast the second input - batch normalization case
        if (in1.shape.size() != 4 || in2.shape.size() != 4)
        {
            throw std::runtime_error("Both inputs must be 4D arrays");
        }

        // Calculate strides for faster access
        const int in1Stride0 = in1.stride[0];
        const int in1Stride1 = in1.stride[1];
        const int in1Stride2 = in1.stride[2];
        const int in1Stride3 = in1.stride[3];

        // Handle broadcasting
        for (int n = 0; n < in1.shape[0]; ++n)
        {
            for (int c = 0; c < in1.shape[1]; ++c)
            { // channels - this contains the addition value on the second input
                for (int h = 0; h < in1.shape[2]; ++h)
                {
                    for (int w = 0; w < in1.shape[3]; ++w)
                    {
                        int in1Idx = n * in1Stride0 + c * in1Stride1 + h * in1Stride2 + w * in1Stride3;
                        in1.data[in1Idx] += in2.data[c];
                    }
                }
            }
        }
    }

    template void Layers::Add_Inplace<int>(types::ArrayND<int> &, const types::ArrayND<int> &);
    template void Layers::Add_Inplace<double>(types::ArrayND<double> &, const types::ArrayND<double> &);
    template void Layers::Add_Inplace<uint8_t>(types::ArrayND<uint8_t> &, const types::ArrayND<uint8_t> &);

    template <typename T>
    types::ArrayND<T> Layers::Mul(
        const types::ArrayND<T> &in1, // {1, channels, height, width }
        const types::ArrayND<T> &in2  // {1, value_per_batch, 1, 1}
    )
    {
        // if they are of the same shape, just multiply them
        if (in1.shape == in2.shape)
        {
            return utility::mul(in1, in2);
        }

        if (in1.shape.size() != 4 || in2.shape.size() != 4)
        {
            throw std::runtime_error("Both inputs must be 4D arrays");
        }

        types::ArrayND<T> output = utility::createZeros<T>(in1.shape);

        // Calculate strides for faster access
        const int in1Stride0 = in1.stride[0];
        const int in1Stride1 = in1.stride[1];
        const int in1Stride2 = in1.stride[2];
        const int in1Stride3 = in1.stride[3];

        // Handle broadcasting
        for (int n = 0; n < in1.shape[0]; ++n)
        {
            for (int c = 0; c < in1.shape[1]; ++c)
            { // channels - this contains the multiplication value on the second input
                for (int h = 0; h < in1.shape[2]; ++h)
                {
                    for (int w = 0; w < in1.shape[3]; ++w)
                    {
                        int in1Idx = n * in1Stride0 + c * in1Stride1 + h * in1Stride2 + w * in1Stride3;
                        output.data[in1Idx] = in1.data[in1Idx] * in2.data[c];
                    }
                }
            }
        }

        return output;
    }

    template types::ArrayND<int> Layers::Mul<int>(const types::ArrayND<int> &, const types::ArrayND<int> &);
    template types::ArrayND<double> Layers::Mul<double>(const types::ArrayND<double> &, const types::ArrayND<double> &);
    template types::ArrayND<uint8_t> Layers::Mul<uint8_t>(const types::ArrayND<uint8_t> &, const types::ArrayND<uint8_t> &);

    template <typename T>
    void Layers::Mul_Inplace(
        types::ArrayND<T> &in1,      // {1, channels, height, width }
        const types::ArrayND<T> &in2 // {1, value_per_batch, 1, 1}
    )
    {
        if (in1.shape == in2.shape)
        {
            return utility::mul_inplace(in1, in2);
        }

        if (in1.shape.size() != 4 || in2.shape.size() != 4)
        {
            throw std::runtime_error("Both inputs must be 4D arrays");
        }

        // Calculate strides for faster access
        const int in1Stride0 = in1.stride[0];
        const int in1Stride1 = in1.stride[1];
        const int in1Stride2 = in1.stride[2];
        const int in1Stride3 = in1.stride[3];

        // Handle broadcasting
        for (int n = 0; n < in1.shape[0]; ++n)
        {
            for (int c = 0; c < in1.shape[1]; ++c)
            { // channels - this contains the multiplication value on the second input
                for (int h = 0; h < in1.shape[2]; ++h)
                {
                    for (int w = 0; w < in1.shape[3]; ++w)
                    {
                        int in1Idx = n * in1Stride0 + c * in1Stride1 + h * in1Stride2 + w * in1Stride3;
                        in1.data[in1Idx] *= in2.data[c];
                    }
                }
            }
        }
    }

    template void Layers::Mul_Inplace<int>(types::ArrayND<int> &, const types::ArrayND<int> &);
    template void Layers::Mul_Inplace<double>(types::ArrayND<double> &, const types::ArrayND<double> &);
    template void Layers::Mul_Inplace<uint8_t>(types::ArrayND<uint8_t> &, const types::ArrayND<uint8_t> &);

    template <typename T>
    void Layers::Mul_And_Add(
        types::ArrayND<T> &in1,
        const types::ArrayND<T> &in2,
        const types::ArrayND<T> &in3)
    {
        // Calculate strides for faster access
        const int in1Stride0 = in1.stride[0];
        const int in1Stride1 = in1.stride[1];
        const int in1Stride2 = in1.stride[2];
        const int in1Stride3 = in1.stride[3];

        // Handle broadcasting
        for (int n = 0; n < in1.shape[0]; ++n)
        {
            for (int c = 0; c < in1.shape[1]; ++c)
            { // channels - this contains the multiplication value on the second input
                for (int h = 0; h < in1.shape[2]; ++h)
                {
                    for (int w = 0; w < in1.shape[3]; ++w)
                    {
                        int in1Idx = n * in1Stride0 + c * in1Stride1 + h * in1Stride2 + w * in1Stride3;
                        in1.data[in1Idx] *= in2.data[c];
                        in1.data[in1Idx] += in3.data[c];
                    }
                }
            }
        }
    }

    template void Layers::Mul_And_Add<double>(
        types::ArrayND<double> &in1,
        const types::ArrayND<double> &in2,
        const types::ArrayND<double> &in3);
}