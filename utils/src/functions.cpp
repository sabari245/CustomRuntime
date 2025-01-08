#include "functions.h"
#include <functional>
#include <numeric>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <iterator>

namespace utility
{

    // ND array creation functions

    template <typename T>
    types::ArrayND<T> createOnes(const std::vector<int> &shape)
    {
        types::ArrayND<T> array;
        array.shape = shape;
        array.data.resize(std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>()), 1);
        array.stride = getStrideFromShape(shape);
        return array;
    }

    template types::ArrayND<double> createOnes(const std::vector<int> &shape);
    template types::ArrayND<int> createOnes(const std::vector<int> &shape);
    template types::ArrayND<uint8_t> createOnes(const std::vector<int> &shape);

    template <typename T>
    types::ArrayND<T> createZeros(const std::vector<int> &shape)
    {
        types::ArrayND<T> array;
        array.shape = shape;
        array.data.resize(std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>()), 0);
        array.stride = getStrideFromShape(shape);
        return array;
    }

    template types::ArrayND<double> createZeros(const std::vector<int> &shape);
    template types::ArrayND<int> createZeros(const std::vector<int> &shape);
    template types::ArrayND<uint8_t> createZeros(const std::vector<int> &shape);

    // ND array accessing functions

    template <typename T>
    T get(const types::ArrayND<T> &array, const std::vector<int> &indices)
    {
        int index = 0;
        for (int i = 0; i < indices.size(); i++)
        {
            index += indices[i] * array.stride[i];
        }
        return array.data[index];
    }

    template double get(const types::ArrayND<double> &array, const std::vector<int> &indices);
    template int get(const types::ArrayND<int> &array, const std::vector<int> &indices);
    template uint8_t get(const types::ArrayND<uint8_t> &array, const std::vector<int> &indices);

    template <typename T>
    types::ArrayND<T> slice(const types::ArrayND<T> &array, int start, int end)
    {
        types::ArrayND<T> slicedArray;

        if (start < 0 || start >= array.shape[0] || end < 0 || end > array.shape[0] || start >= end)
        {
            throw std::runtime_error("Invalid slice indices");
        }

        slicedArray.shape = {end - start};

        std::copy(array.shape.begin() + 1, array.shape.end(), std::back_inserter(slicedArray.shape));

        slicedArray.stride = getStrideFromShape(slicedArray.shape);

        std::copy(array.data.begin() + start * array.stride[0], array.data.begin() + end * array.stride[0], std::back_inserter(slicedArray.data));

        return slicedArray;
    }

    template types::ArrayND<double> slice(const types::ArrayND<double> &array, int start, int end);
    template types::ArrayND<int> slice(const types::ArrayND<int> &array, int start, int end);
    template types::ArrayND<uint8_t> slice(const types::ArrayND<uint8_t> &array, int start, int end);

    template <typename T>
    types::ArrayND<T> sliceND(const types::ArrayND<T> &array,
                              const std::vector<int> &start,
                              const std::vector<int> &end)
    {
        // Validate input dimensions
        if (start.size() != end.size())
        {
            throw std::runtime_error("Start and end must have same number of dimensions");
        }
        if (start.size() > array.shape.size())
        {
            throw std::runtime_error("Too many slice dimensions specified");
        }

        types::ArrayND<T> slicedArray;

        // Validate indices and calculate new shape
        slicedArray.shape.reserve(array.shape.size());
        for (size_t dim = 0; dim < start.size(); dim++)
        {
            if (start[dim] < 0 || start[dim] >= array.shape[dim] ||
                end[dim] < 0 || end[dim] > array.shape[dim] ||
                start[dim] >= end[dim])
            {
                throw std::runtime_error("Invalid slice indices at dimension " + std::to_string(dim));
            }
            slicedArray.shape.push_back(end[dim] - start[dim]);
        }

        // Copy remaining dimensions that weren't sliced
        for (size_t dim = start.size(); dim < array.shape.size(); dim++)
        {
            slicedArray.shape.push_back(array.shape[dim]);
        }

        // Calculate new strides
        slicedArray.stride = getStrideFromShape(slicedArray.shape);

        // Calculate total elements in slice
        size_t totalElements = 1;
        for (size_t dim = 0; dim < slicedArray.shape.size(); dim++)
        {
            totalElements *= slicedArray.shape[dim];
        }

        // Helper function to recursively iterate through dimensions
        std::function<void(size_t, size_t, size_t)> copySlice =
            [&](size_t dimIndex, size_t srcIndex, size_t dstIndex)
        {
            if (dimIndex == start.size())
            {
                // Copy the contiguous block at this position
                size_t blockSize = 1;
                for (size_t i = dimIndex; i < array.shape.size(); i++)
                {
                    blockSize *= array.shape[i];
                }
                std::copy(
                    array.data.begin() + srcIndex,
                    array.data.begin() + srcIndex + blockSize,
                    slicedArray.data.begin() + dstIndex);
                return;
            }

            // Iterate through current dimension
            size_t srcStride = array.stride[dimIndex];
            size_t dstStride = slicedArray.stride[dimIndex];
            for (int i = start[dimIndex], j = 0; i < end[dimIndex]; i++, j++)
            {
                copySlice(dimIndex + 1,
                          srcIndex + i * srcStride,
                          dstIndex + j * dstStride);
            }
        };

        // Preallocate the data vector instead of using back_inserter
        slicedArray.data.resize(totalElements);

        // Start recursive copy from first dimension
        copySlice(0, 0, 0);

        return slicedArray;
    }

    template types::ArrayND<double> sliceND(const types::ArrayND<double> &array, const std::vector<int> &start, const std::vector<int> &end);
    template types::ArrayND<int> sliceND(const types::ArrayND<int> &array, const std::vector<int> &start, const std::vector<int> &end);
    template types::ArrayND<uint8_t> sliceND(const types::ArrayND<uint8_t> &array, const std::vector<int> &start, const std::vector<int> &end);

    template <typename T>
    types::ArrayND<T> reshape(const types::ArrayND<T> &array, const std::vector<int> &newShape)
    {
        // Validate input
        int total_elements = array.data.size();
        int newShape_product = 1;
        int negative_one_count = 0;
        int inferred_dimension = -1;

        for (size_t i = 0; i < newShape.size(); ++i)
        {
            if (newShape[i] == -1)
            {
                negative_one_count++;
                inferred_dimension = i;
            }
            else
            {
                if (newShape[i] <= 0)
                {
                    throw std::invalid_argument("Shape dimensions must be positive or -1");
                }
                newShape_product *= newShape[i];
            }
        }

        // Ensure only one dimension is marked as -1
        if (negative_one_count > 1)
        {
            throw std::invalid_argument("Only one dimension can be inferred (-1)");
        }

        // Infer the dimension marked as -1
        std::vector<int> finalShape = newShape;
        if (negative_one_count == 1)
        {
            if (total_elements % newShape_product != 0)
            {
                throw std::invalid_argument("Total size of elements is not divisible to infer the -1 dimension");
            }
            finalShape[inferred_dimension] = total_elements / newShape_product;
        }
        else if (newShape_product != total_elements)
        {
            throw std::invalid_argument("New shape must have the same total number of elements as the original array");
        }

        // Create the reshaped array
        types::ArrayND<T> reshapedArray;
        reshapedArray.shape = finalShape;
        reshapedArray.stride = getStrideFromShape(finalShape);
        reshapedArray.data = array.data;

        return reshapedArray;
    }

    template types::ArrayND<double> reshape(const types::ArrayND<double> &array, const std::vector<int> &newShape);
    template types::ArrayND<int> reshape(const types::ArrayND<int> &array, const std::vector<int> &newShape);
    template types::ArrayND<uint8_t> reshape(const types::ArrayND<uint8_t> &array, const std::vector<int> &newShape);

    template <typename T>
    void reshape_inplace(types::ArrayND<T> &array, const std::vector<int> &newShape)
    {
        // Validate input
        int total_elements = array.data.size();
        int newShape_product = 1;
        int negative_one_count = 0;
        int inferred_dimension = -1;

        // Check dimensions and calculate product
        for (size_t i = 0; i < newShape.size(); ++i)
        {
            if (newShape[i] == -1)
            {
                negative_one_count++;
                inferred_dimension = i;
            }
            else
            {
                if (newShape[i] <= 0)
                {
                    throw std::invalid_argument("Shape dimensions must be positive or -1");
                }
                newShape_product *= newShape[i];
            }
        }

        // Ensure only one dimension is marked as -1
        if (negative_one_count > 1)
        {
            throw std::invalid_argument("Only one dimension can be inferred (-1)");
        }

        // Infer the dimension marked as -1
        std::vector<int> finalShape = newShape;
        if (negative_one_count == 1)
        {
            if (total_elements % newShape_product != 0)
            {
                throw std::invalid_argument("Total size of elements is not divisible to infer the -1 dimension");
            }
            finalShape[inferred_dimension] = total_elements / newShape_product;
        }
        else if (newShape_product != total_elements)
        {
            throw std::invalid_argument("New shape must have the same total number of elements as the original array");
        }

        // Modify the array in place
        array.shape = finalShape;
        array.stride = getStrideFromShape(finalShape);
    }

    template void reshape_inplace(types::ArrayND<double> &array, const std::vector<int> &newShape);
    template void reshape_inplace(types::ArrayND<int> &array, const std::vector<int> &newShape);
    template void reshape_inplace(types::ArrayND<uint8_t> &array, const std::vector<int> &newShape);

    template <typename T>
    types::ArrayND<T> addPadding(const types::ArrayND<T> &array, const std::vector<int> &padding)
    {
        if (array.shape.size() < 2)
        {
            throw std::runtime_error("Array must have at least 2 dimensions");
        }
        if (padding.size() != 4)
        {
            throw std::runtime_error("Padding must contain exactly 4 values [top, right, bottom, left]");
        }

        // Modify only the last two dimensions (height and width)
        const size_t height_dim = array.shape.size() - 2;
        const size_t width_dim = array.shape.size() - 1;

        std::vector<int> paddedShape = array.shape;
        paddedShape[height_dim] += (padding[0] + padding[2]);
        paddedShape[width_dim] += (padding[1] + padding[3]);

        types::ArrayND<T> paddedArray = createZeros<T>(paddedShape);

        // Calculate the product of dimensions before height (batch size * channels * ...)
        size_t outer_dims_product = 1;
        for (size_t i = 0; i < height_dim; ++i)
        {
            outer_dims_product *= array.shape[i];
        }

        std::vector<size_t> idx(array.shape.size(), 0);

        for (size_t outer = 0; outer < outer_dims_product; ++outer)
        {
            size_t temp = outer;
            for (size_t i = 0; i < height_dim; ++i)
            {
                idx[i] = temp % array.shape[i];
                temp /= array.shape[i];
            }

            // Iterate over height and width
            for (int h = 0; h < array.shape[height_dim]; ++h)
            {
                for (int w = 0; w < array.shape[width_dim]; ++w)
                {
                    idx[height_dim] = h;
                    idx[width_dim] = w;

                    size_t src_idx = 0;
                    for (size_t i = 0; i < array.shape.size(); ++i)
                    {
                        src_idx += idx[i] * array.stride[i];
                    }

                    idx[height_dim] = h + padding[0];
                    idx[width_dim] = w + padding[2];
                    size_t dst_idx = 0;
                    for (size_t i = 0; i < paddedArray.shape.size(); ++i)
                    {
                        dst_idx += idx[i] * paddedArray.stride[i];
                    }

                    paddedArray.data[dst_idx] = array.data[src_idx];
                }
            }
        }

        return paddedArray;
    }

    template types::ArrayND<double> addPadding(const types::ArrayND<double> &array, const std::vector<int> &padding);
    template types::ArrayND<int> addPadding(const types::ArrayND<int> &array, const std::vector<int> &padding);
    template types::ArrayND<uint8_t> addPadding(const types::ArrayND<uint8_t> &array, const std::vector<int> &padding);

    template <typename T>
    types::ArrayND<T> transpose(const types::ArrayND<T> &array, const std::vector<int> &perm)
    {
        // Validate input
        if (array.shape.size() != perm.size())
        {
            throw std::invalid_argument("Permutation size must match array dimensions");
        }

        types::ArrayND<T> result;

        // Rearrange shape according to permutation
        result.shape.resize(array.shape.size());
        for (size_t i = 0; i < perm.size(); i++)
        {
            result.shape[i] = array.shape[perm[i]];
        }

        // Calculate new strides
        result.stride = getStrideFromShape(result.shape);

        // Initialize output data
        size_t total_size = std::accumulate(array.shape.begin(), array.shape.end(),
                                            static_cast<size_t>(1), std::multiplies<size_t>());
        result.data.resize(total_size);

        // Create index arrays for iteration
        std::vector<int> current_idx(array.shape.size(), 0);

        // Iterate through all elements
        for (size_t flat_idx = 0; flat_idx < total_size; flat_idx++)
        {
            // Calculate source index
            size_t src_idx = 0;
            for (size_t dim = 0; dim < array.shape.size(); dim++)
            {
                src_idx += current_idx[dim] * array.stride[dim];
            }

            // Calculate destination index
            size_t dst_idx = 0;
            for (size_t dim = 0; dim < result.shape.size(); dim++)
            {
                dst_idx += current_idx[perm[dim]] * result.stride[dim];
            }

            // Copy data
            result.data[dst_idx] = array.data[src_idx];

            // Update indices
            for (int dim = array.shape.size() - 1; dim >= 0; dim--)
            {
                current_idx[dim]++;
                if (current_idx[dim] < array.shape[dim])
                {
                    break;
                }
                current_idx[dim] = 0;
            }
        }

        return result;
    }

    template types::ArrayND<double> transpose(const types::ArrayND<double> &array, const std::vector<int> &perm);
    template types::ArrayND<int> transpose(const types::ArrayND<int> &array, const std::vector<int> &perm);
    template types::ArrayND<uint8_t> transpose(const types::ArrayND<uint8_t> &array, const std::vector<int> &perm);

    template <typename T>
    types::ArrayND<T> flatten(const types::ArrayND<T> &array)
    {
        types::ArrayND<T> flattenedArray;
        flattenedArray.shape = {std::accumulate(array.shape.begin(), array.shape.end(), 1, std::multiplies<int>())};
        flattenedArray.data = array.data;
        flattenedArray.stride = {1};
        return flattenedArray;
    }

    template types::ArrayND<double> flatten(const types::ArrayND<double> &array);
    template types::ArrayND<int> flatten(const types::ArrayND<int> &array);
    template types::ArrayND<uint8_t> flatten(const types::ArrayND<uint8_t> &array);

    template <typename T>
    types::ArrayND<T> add(const types::ArrayND<T> &array1, const types::ArrayND<T> &array2)
    {
        if (array1.shape != array2.shape)
        {
            throw std::runtime_error("Array shapes must match for addition");
        }
        types::ArrayND<T> result;
        result.shape = array1.shape;
        result.data.resize(array1.data.size());
        std::transform(array1.data.begin(), array1.data.end(), array2.data.begin(), result.data.begin(), std::plus<T>());
        return result;
    }

    template types::ArrayND<double> add(const types::ArrayND<double> &array1, const types::ArrayND<double> &array2);
    template types::ArrayND<int> add(const types::ArrayND<int> &array1, const types::ArrayND<int> &array2);
    template types::ArrayND<uint8_t> add(const types::ArrayND<uint8_t> &array1, const types::ArrayND<uint8_t> &array2);

    template <typename T>
    types::ArrayND<T> add(const types::ArrayND<T> &array, T scalar)
    {
        types::ArrayND<T> result;
        result.shape = array.shape;
        result.data.resize(array.data.size());
        result.stride = array.stride;
        std::transform(array.data.begin(), array.data.end(), result.data.begin(), [scalar](T val)
                       { return val + scalar; });
        return result;
    }

    template types::ArrayND<double> add(const types::ArrayND<double> &array, double scalar);
    template types::ArrayND<int> add(const types::ArrayND<int> &array, int scalar);
    template types::ArrayND<uint8_t> add(const types::ArrayND<uint8_t> &array, uint8_t scalar);

    template <typename T>
    void add_inplace(types::ArrayND<T> &array1, const types::ArrayND<T> &array2)
    {
        if (array1.shape != array2.shape)
        {
            throw std::runtime_error("Array shapes must match for addition");
        }
        std::transform(array1.data.begin(), array1.data.end(), array2.data.begin(), array1.data.begin(), std::plus<T>());
    }

    template void add_inplace(types::ArrayND<double> &array1, const types::ArrayND<double> &array2);
    template void add_inplace(types::ArrayND<int> &array1, const types::ArrayND<int> &array2);
    template void add_inplace(types::ArrayND<uint8_t> &array1, const types::ArrayND<uint8_t> &array2);

    template <typename T>
    types::ArrayND<T> mul(const types::ArrayND<T> &array1, const types::ArrayND<T> &array2)
    {
        if (array1.shape != array2.shape)
        {
            throw std::runtime_error("Array shapes must match for multiplication");
        }
        types::ArrayND<T> result;
        result.shape = array1.shape;
        result.data.resize(array1.data.size());
        std::transform(array1.data.begin(), array1.data.end(), array2.data.begin(), result.data.begin(), std::multiplies<T>());
        return result;
    }

    template types::ArrayND<double> mul(const types::ArrayND<double> &array1, const types::ArrayND<double> &array2);
    template types::ArrayND<int> mul(const types::ArrayND<int> &array1, const types::ArrayND<int> &array2);
    template types::ArrayND<uint8_t> mul(const types::ArrayND<uint8_t> &array1, const types::ArrayND<uint8_t> &array2);

    template <typename T>
    void mul_inplace(types::ArrayND<T> &array1, const types::ArrayND<T> &array2)
    {
        if (array1.shape != array2.shape)
        {
            throw std::runtime_error("Array shapes must match for multiplication");
        }
        std::transform(array1.data.begin(), array1.data.end(), array2.data.begin(), array1.data.begin(), std::multiplies<T>());
    }

    template void mul_inplace(types::ArrayND<double> &array1, const types::ArrayND<double> &array2);
    template void mul_inplace(types::ArrayND<int> &array1, const types::ArrayND<int> &array2);
    template void mul_inplace(types::ArrayND<uint8_t> &array1, const types::ArrayND<uint8_t> &array2);

    // Miscilaneous functions

    std::vector<int> getStrideFromShape(const std::vector<int> &shape)
    {
        std::vector<int> stride(shape.size(), 1);
        for (int i = shape.size() - 2; i >= 0; i--)
        {
            stride[i] = shape[i + 1] * stride[i + 1];
        }
        return stride;
    }

    template <typename T>
    int getOffset(const types::ArrayND<T> &array, const std::vector<int> &indices)
    {
        if (indices.size() != array.shape.size())
        {
            throw std::runtime_error("Invalid number of indices");
        }
        int offset = 0;
        for (size_t i = 0; i < indices.size(); i++)
        {

            if (indices[i] < 0 || indices[i] >= array.shape[i])
            {
                throw std::runtime_error("Index out of bounds");
            }

            offset += indices[i] * array.stride[i];
        }
        return offset;
    }

    template int getOffset(const types::ArrayND<int> &array, const std::vector<int> &indices);
    template int getOffset(const types::ArrayND<double> &array, const std::vector<int> &indices);
    template int getOffset(const types::ArrayND<uint8_t> &array, const std::vector<int> &indices);

    template <typename T>
    types::ArrayND<T> convertUint8ToType(const types::ArrayND<uint8_t> &array)
    {
        types::ArrayND<T> convertedArray;
        convertedArray.shape = array.shape;
        convertedArray.stride = array.stride;
        convertedArray.data.resize(array.data.size());
        std::transform(array.data.begin(), array.data.end(), convertedArray.data.begin(), [](uint8_t val)
                       { return static_cast<T>(val); });
        return convertedArray;
    }

    template types::ArrayND<double> convertUint8ToType(const types::ArrayND<uint8_t> &array);
    template types::ArrayND<int> convertUint8ToType(const types::ArrayND<uint8_t> &array);

}