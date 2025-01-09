#include <gtest/gtest.h>

#include "types.h"
#include "layers.h"
#include "functions.h"

TEST(MaxPool, test1)
{
	types::ArrayND<int> array;
	array.shape = {1, 2, 3, 3};
	array.data = {
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
		10, 11, 12,
		13, 14, 15,
		16, 17, 18};

	array.stride = utility::getStrideFromShape(array.shape);

	layers::Layers engine;

	auto result = engine.maxPool(array, {2, 2}, {1, 1});

	std::vector<int> expected = {
		5, 6,
		8, 9,
		14, 15,
		17, 18};

	for (int i = 0; i < result.data.size(); i++)
	{
		EXPECT_EQ(result.data[i], expected[i]);
	}

	result = engine.maxPool(array, {2, 2}, {2, 2});

	std::vector<int> expected2 = {
		5, 14};

	for (int i = 0; i < result.data.size(); i++)
	{
		EXPECT_EQ(result.data[i], expected2[i]);
	}
}

TEST(ConvolutionLayer, Test1)
{
	types::ArrayND<int> in;
	in.shape = {1, 2, 3, 3};
	in.data = {
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
		10, 11, 12,
		13, 14, 15,
		16, 17, 18};
	in.stride = utility::getStrideFromShape(in.shape);

	types::ArrayND<int> kernel;
	kernel.shape = {1, 2, 2, 2};
	kernel.data = {
		1, 0,
		0, 1,
		0, 1,
		1, 0};
	kernel.stride = utility::getStrideFromShape(kernel.shape);

	types::ArrayND<int> bias;
	bias.shape = {2};
	bias.data = {1, 1};
	bias.stride = utility::getStrideFromShape(bias.shape);

	layers::Layers engine;

	auto result = engine.conv2d(in, kernel, bias, {0, 0, 0, 0}, {1, 1}, {2, 2});

	std::vector<int> expected = {
		30, 34,
		42, 46};

	for (int i = 0; i < result.data.size(); i++)
	{
		EXPECT_EQ(result.data[i], expected[i] + 1);
	}
}

TEST(FFTOperation, Test1)
{
	types::ArrayND<int> in;
	in.shape = {4, 4};
	in.data = {
		0, 1, 2, 3,
		4, 5, 6, 7,
		8, 9, 10, 11,
		12, 13, 14, 15};
	in.stride = utility::getStrideFromShape(in.shape);

	types::ArrayND<int> kernel;
	kernel.shape = {2, 2};
	kernel.data = {
		1, 0,
		0, 1};
	kernel.stride = utility::getStrideFromShape(kernel.shape);

	layers::Layers engine;

	auto result = engine.convolveFFT(in, kernel);

	std::vector<int> expected = {5, 7, 9, 13, 15, 17, 21, 23, 25};
	for (int i = 0; i < expected.size(); i++)
	{
		EXPECT_EQ(result.data[i], expected[i]);
	}
}

TEST(FFTOperation, Test2)
{
	types::ArrayND<int> in;
	in.shape = {3, 3};
	in.data = {
		1, 2, 3,
		4, 5, 6,
		7, 8, 9};
	in.stride = utility::getStrideFromShape(in.shape);

	types::ArrayND<int> kernel;
	kernel.shape = {2, 2};
	kernel.data = {
		1, 0,
		0, 1};
	kernel.stride = utility::getStrideFromShape(kernel.shape);

	layers::Layers engine;

	auto result = engine.convolveFFT(in, kernel);

	std::vector<int> expected = {6, 8, 12, 14};
	for (int i = 0; i < expected.size(); i++)
	{
		EXPECT_EQ(result.data[i], expected[i]);
	}
}

TEST(ConvolutionLayer, Test2)
{
	types::ArrayND<int> in;
	in.shape = {1, 2, 3, 3};
	in.data = {
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
		10, 11, 12,
		13, 14, 15,
		16, 17, 18};
	in.stride = utility::getStrideFromShape(in.shape);

	types::ArrayND<int> kernel;
	kernel.shape = {1, 2, 2, 2};
	kernel.data = {
		1, 0,
		0, 1,
		0, 1,
		1, 0};
	kernel.stride = utility::getStrideFromShape(kernel.shape);

	types::ArrayND<int> bias;
	bias.shape = {2};
	bias.data = {1, 1};
	bias.stride = utility::getStrideFromShape(bias.shape);

	layers::Layers engine;

	auto result = engine.conv2dFFT(in, kernel, bias, {0, 0, 0, 0}, {1, 1}, {2, 2});

	std::vector<int> expected = {
		30, 34,
		42, 46};

	for (int i = 0; i < result.data.size(); i++)
	{
		EXPECT_EQ(result.data[i], expected[i] + 1);
	}
}

TEST(MatMul, Test1)
{
	types::ArrayND<int> in1;
	in1.shape = {1, 4};
	in1.data = {1, 2, 3, 4};
	in1.stride = utility::getStrideFromShape(in1.shape);

	types::ArrayND<int> in2;
	in2.shape = {4, 2};
	in2.data = {1, 2, 3, 4, 5, 6, 7, 8};
	in2.stride = utility::getStrideFromShape(in2.shape);

	layers::Layers engine;
	auto result = engine.MatMul(in1, in2);

	std::vector<int> expected = {50, 60};

	for (int i = 0; i < result.data.size(); i++)
	{
		EXPECT_EQ(result.data[i], expected[i]);
	}
}

TEST(AdditionLayer, test1)
{
	types::ArrayND<int> in1;
	in1.shape = {10};
	in1.data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
	in1.stride = utility::getStrideFromShape(in1.shape);

	in1 = utility::reshape(in1, {1, 10});

	types::ArrayND<int> in2;
	in2.shape = {10};
	in2.data = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
	in2.stride = utility::getStrideFromShape(in2.shape);

	in2 = utility::reshape(in2, {1, 10});

	layers::Layers engine;
	auto result = engine.Add(in1, in2);

	std::vector<int> expected = {11, 11, 11, 11, 11, 11, 11, 11, 11, 11};

	for (int i = 0; i < result.data.size(); i++)
	{
		EXPECT_EQ(result.data[i], expected[i]);
	}
}

TEST(AdditionLayer, test2)
{
	types::ArrayND<int> in1;
	in1.shape = {1, 2, 2, 2};
	in1.data = {
		1, 2,
		3, 4,

		5, 6,
		7, 8};
	in1.stride = utility::getStrideFromShape(in1.shape);

	types::ArrayND<int> in2;
	in2.shape = {1, 2, 1, 1};
	in2.data = {2, 3};
	in2.stride = utility::getStrideFromShape(in2.shape);

	layers::Layers engine;
	auto result = engine.Add(in1, in2);

	std::vector<int> expected = {3, 4, 5, 6, 8, 9, 10, 11};

	for (int i = 0; i < result.data.size(); i++)
	{
		EXPECT_EQ(result.data[i], expected[i]);
	}
}

TEST(MultiplicationLayer, Test1)
{
	types::ArrayND<int> in1;
	in1.shape = {10};
	in1.data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
	in1.stride = utility::getStrideFromShape(in1.shape);

	in1 = utility::reshape(in1, {1, 10});

	types::ArrayND<int> in2;
	in2.shape = {10};
	in2.data = {2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
	in2.stride = utility::getStrideFromShape(in2.shape);

	in2 = utility::reshape(in2, {1, 10});

	layers::Layers engine;
	auto result = engine.Mul(in1, in2);

	std::vector<int> expected = {2, 4, 6, 8, 10, 12, 14, 16, 18, 20};

	for (int i = 0; i < result.data.size(); i++)
	{
		EXPECT_EQ(result.data[i], expected[i]);
	}
}

TEST(MultiplicationLayer, Test2)
{
	types::ArrayND<int> in1;
	in1.shape = {1, 2, 2, 2};
	in1.data = {
		1, 2,
		3, 4,
		5, 6,
		7, 8};
	in1.stride = utility::getStrideFromShape(in1.shape);
	types::ArrayND<int> in2;
	in2.shape = {1, 2, 1, 1};
	in2.data = {2, 3};
	in2.stride = utility::getStrideFromShape(in2.shape);

	layers::Layers engine;
	auto result = engine.Mul(in1, in2);

	std::vector<int> expected = {2, 4, 6, 8, 15, 18, 21, 24};

	for (int i = 0; i < result.data.size(); i++)
	{
		EXPECT_EQ(result.data[i], expected[i]);
	}
}