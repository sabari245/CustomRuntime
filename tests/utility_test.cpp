#include <gtest/gtest.h>

#include "types.h"
#include "functions.h"

types::ArrayND<int> createTestingArray()
{
	types::ArrayND<int> array;
	array.shape = {2, 3, 4};
	array.data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
	array.stride = utility::getStrideFromShape(array.shape);
	return array;
}

TEST(CreateFunctions, CreateOnes)
{
	std::vector<int> shape = {2, 3, 4};
	auto array = utility::createOnes<double>(shape);

	int totalElements = 2 * 3 * 4, i = 0;

	for (auto val : array.data)
	{
		EXPECT_EQ(val, 1);
		i++;
	}

	EXPECT_EQ(i, totalElements);
}

TEST(CreateFunctions, CreateZeros)
{
	std::vector<int> shape = {2, 3, 4};
	auto array = utility::createZeros<double>(shape);
	int totalElements = 2 * 3 * 4, i = 0;
	for (auto val : array.data)
	{
		EXPECT_EQ(val, 0);
		i++;
	}
	EXPECT_EQ(i, totalElements);
}

TEST(AccessFunctions, Get)
{
	std::vector<int> shape = {2, 3, 4};
	types::ArrayND<double> array;
	array.shape = shape;
	array.data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
	array.stride = utility::getStrideFromShape(shape);
	EXPECT_EQ(utility::get(array, {0, 0, 0}), 1);
	EXPECT_EQ(utility::get(array, {0, 0, 1}), 2);
	EXPECT_EQ(utility::get(array, {0, 1, 0}), 5);
	EXPECT_EQ(utility::get(array, {1, 0, 0}), 13);
	EXPECT_EQ(utility::get(array, {1, 2, 3}), 24);
}

TEST(AccessFunctions, Slice)
{
	types::ArrayND<int> array = createTestingArray();

	auto slicedArray = utility::slice(array, 0, 1);

	EXPECT_EQ(slicedArray.shape[0], 1);
	EXPECT_EQ(slicedArray.shape[1], 3);
	EXPECT_EQ(slicedArray.shape[2], 4);

	std::vector<int> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

	for (int i = 0; i < slicedArray.data.size(); i++)
	{
		EXPECT_EQ(slicedArray.data[i], data[i]);
	}
}

TEST(AccessFunctions, SliceND)
{
	types::ArrayND<int> array = createTestingArray();
	auto slicedArray = utility::sliceND(array, {0, 0, 0}, {1, 2, 3});
	EXPECT_EQ(slicedArray.shape[0], 1);
	EXPECT_EQ(slicedArray.shape[1], 2);
	EXPECT_EQ(slicedArray.shape[2], 3);
	std::vector<int> data = {1, 2, 3, 5, 6, 7};

	EXPECT_EQ(slicedArray.data.size(), data.size());

	for (int i = 0; i < slicedArray.data.size(); i++)
	{
		EXPECT_EQ(slicedArray.data[i], data[i]);
	}
}
TEST(AccessFunctions, SliceND2)
{
	types::ArrayND<int> array = createTestingArray();
	auto slicedArray = utility::sliceND(array, {0, 0, 0}, {1, 2, 3});

	auto reshapedArray = utility::reshape(slicedArray, {2, 3});
	EXPECT_EQ(reshapedArray.shape[0], 2);
	EXPECT_EQ(reshapedArray.shape[1], 3);
	std::vector<int> data = {1, 2, 3, 5, 6, 7};

	EXPECT_EQ(reshapedArray.data.size(), data.size());

	for (int i = 0; i < reshapedArray.data.size(); i++)
	{
		EXPECT_EQ(reshapedArray.data[i], data[i]);
	}
}

TEST(AccessFunctions, SliceND3)
{
	types::ArrayND<int> array = createTestingArray();
	auto slicedArray = utility::sliceND(array, {0, 0, 0}, {1, 2, 3});

	utility::reshape_inplace(slicedArray, {2, 3});
	EXPECT_EQ(slicedArray.shape[0], 2);
	EXPECT_EQ(slicedArray.shape[1], 3);
	std::vector<int> data = {1, 2, 3, 5, 6, 7};

	EXPECT_EQ(slicedArray.data.size(), data.size());

	for (int i = 0; i < slicedArray.data.size(); i++)
	{
		EXPECT_EQ(slicedArray.data[i], data[i]);
	}
}

TEST(AccessFunctions, SliceND4)
{
	types::ArrayND<int> array = createTestingArray();
	auto slicedArray = utility::sliceND(array, {0, 0, 0}, {2, 3, 4});

	EXPECT_EQ(slicedArray.shape[0], 2);
	EXPECT_EQ(slicedArray.shape[1], 3);
	EXPECT_EQ(slicedArray.shape[2], 4);

	EXPECT_EQ(slicedArray.data.size(), array.data.size());

	for (int i = 0; i < slicedArray.data.size(); i++)
	{
		EXPECT_EQ(slicedArray.data[i], array.data[i]);
	}
}

TEST(ManipulationFunctions, Reshape)
{
	types::ArrayND<int> array = createTestingArray();
	auto reshapedArray = utility::reshape(array, {3, 2, 4});
	EXPECT_EQ(reshapedArray.shape[0], 3);
	EXPECT_EQ(reshapedArray.shape[1], 2);
	EXPECT_EQ(reshapedArray.shape[2], 4);
	std::vector<int> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
	for (int i = 0; i < reshapedArray.data.size(); i++)
	{
		EXPECT_EQ(reshapedArray.data[i], data[i]);
	}

	reshapedArray.data[4] = 100;
	EXPECT_NE(reshapedArray.data[4], array.data[4]);
}

TEST(ManipulationFunctions, Reshape2)
{
	// Create a testing array
	types::ArrayND<int> array = createTestingArray();
	// Total elements = 24, as created in createTestingArray()

	// Test case for -1 to infer a single dimension
	auto reshapedArray = utility::reshape(array, {-1, 4, 3}); // Inferred dimension = 2
	EXPECT_EQ(reshapedArray.shape[0], 2);					  // Total elements / (4 * 3) = 2
	EXPECT_EQ(reshapedArray.shape[1], 4);
	EXPECT_EQ(reshapedArray.shape[2], 3);

	// Ensure data remains the same after reshaping
	std::vector<int> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
							 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
	for (int i = 0; i < reshapedArray.data.size(); i++)
	{
		EXPECT_EQ(reshapedArray.data[i], data[i]);
	}

	// Ensure modifying reshaped array does not modify the original data
	reshapedArray.data[10] = 100;
	EXPECT_NE(reshapedArray.data[10], array.data[10]);

	// Test invalid cases for -1 handling
	EXPECT_THROW(utility::reshape(array, {-1, -1}), std::invalid_argument); // More than one -1
	EXPECT_THROW(utility::reshape(array, {5, -1}), std::invalid_argument);	// Total elements not divisible

	// Test reshaping with only one inferred dimension
	auto reshapedArray2 = utility::reshape(array, {6, -1}); // Infer second dimension
	EXPECT_EQ(reshapedArray2.shape[0], 6);
	EXPECT_EQ(reshapedArray2.shape[1], 4); // Total elements / 6 = 4

	// Ensure data remains intact for reshapedArray2
	for (int i = 0; i < reshapedArray2.data.size(); i++)
	{
		EXPECT_EQ(reshapedArray2.data[i], data[i]);
	}
}

TEST(ManipulationFunctions, Transpose_Case1)
{
	types::ArrayND<int> array = createTestingArray();
	auto transposedArray = utility::transpose(array, {1, 0, 2});

	EXPECT_EQ(transposedArray.shape[0], 3);
	EXPECT_EQ(transposedArray.shape[1], 2);
	EXPECT_EQ(transposedArray.shape[2], 4);

	std::vector<int> data = {1, 2, 3, 4, 13, 14, 15, 16, 5, 6, 7, 8, 17, 18, 19, 20, 9, 10, 11, 12, 21, 22, 23, 24};

	for (int i = 0; i < transposedArray.data.size(); i++)
	{
		EXPECT_EQ(transposedArray.data[i], data[i]);
	}
}

TEST(ManipulationFunction, Transpose_Case2)
{
	types::ArrayND<int> array = createTestingArray();
	auto transposedArray = utility::transpose(array, {2, 1, 0});

	std::vector<int> data = {1, 13, 5, 17, 9, 21, 2, 14, 6, 18, 10, 22, 3, 15, 7, 19, 11, 23, 4, 16, 8, 20, 12, 24};

	for (int i = 0; i < transposedArray.data.size(); i++)
	{
		EXPECT_EQ(transposedArray.data[i], data[i]);
	}
}

TEST(ManipulationFunction, Transpose_case3)
{
	types::ArrayND<int> array = createTestingArray();
	auto transposedArray = utility::transpose(array, {2, 0, 1});

	std::vector<int> data = {1, 5, 9, 13, 17, 21, 2, 6, 10, 14, 18, 22, 3, 7, 11, 15, 19, 23, 4, 8, 12, 16, 20, 24};

	for (int i = 0; i < transposedArray.data.size(); i++)
	{
		EXPECT_EQ(transposedArray.data[i], data[i]);
	}
}

TEST(ManipulationFunctions, padding_test)
{
	types::ArrayND<int> array;
	array.shape = {4, 4};
	array.data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
	array.stride = utility::getStrideFromShape(array.shape);

	auto paddedArray = utility::addPadding(array, {1, 1, 1, 1});

	std::vector<int> expected = {0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 0, 0, 5, 6, 7, 8, 0, 0, 9, 10, 11, 12, 0, 0, 13, 14, 15, 16, 0, 0, 0, 0, 0, 0, 0};

	for (int i = 0; i < paddedArray.data.size(); i++)
	{
		EXPECT_EQ(paddedArray.data[i], expected[i]);
	}

	EXPECT_EQ(paddedArray.shape[0], 6);
	EXPECT_EQ(paddedArray.shape[1], 6);
}

TEST(ManipulationFunctions, padding_test2)
{
	types::ArrayND<int> array;
	array.shape = {4, 4};
	array.data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
	array.stride = utility::getStrideFromShape(array.shape);

	auto paddedArray = utility::addPadding(array, {0, 0, 0, 0});

	std::vector<int> expected = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

	for (int i = 0; i < paddedArray.data.size(); i++)
	{
		EXPECT_EQ(paddedArray.data[i], expected[i]);
	}

	EXPECT_EQ(paddedArray.shape[0], 4);
	EXPECT_EQ(paddedArray.shape[1], 4);
}

TEST(ManipulationFunctions, padding_test3)
{
	types::ArrayND<int> array;
	array.shape = {2, 2, 2};
	array.data = {1, 2, 3, 4, 5, 6, 7, 8};

	array.stride = utility::getStrideFromShape(array.shape);

	auto paddedArray = utility::addPadding(array, {1, 1, 1, 1});

	std::vector<int> expected = {0, 0, 0, 0, 0, 1, 2, 0, 0, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 6, 0, 0, 7, 8, 0, 0, 0, 0, 0};

	for (int i = 0; i < paddedArray.data.size(); i++)
	{
		EXPECT_EQ(paddedArray.data[i], expected[i]);
	}

	EXPECT_EQ(paddedArray.shape[0], 2);
	EXPECT_EQ(paddedArray.shape[1], 4);
	EXPECT_EQ(paddedArray.shape[2], 4);
}

TEST(ManipulationFunctions, addition_test)
{
	types::ArrayND<int> array = createTestingArray();
	auto result = utility::add(array, 10);

	for (int i = 0; i < result.data.size(); i++)
	{
		EXPECT_EQ(result.data[i], array.data[i] + 10);
	}
}

TEST(ManipulationFunctions, addition_test2)
{
	types::ArrayND<int> array1 = createTestingArray();
	types::ArrayND<int> array2 = createTestingArray();

	auto result = utility::add(array1, array2);

	for (int i = 0; i < result.data.size(); i++)
	{
		EXPECT_EQ(result.data[i], array1.data[i] + array2.data[i]);
	}
}

TEST(ManipulationFunctions, multiplication_test)
{
	types::ArrayND<int> array1 = createTestingArray();
	types::ArrayND<int> array2 = createTestingArray();

	auto result = utility::mul(array1, array2);

	for (int i = 0; i < result.data.size(); i++)
	{
		EXPECT_EQ(result.data[i], array1.data[i] * array2.data[i]);
	}
}