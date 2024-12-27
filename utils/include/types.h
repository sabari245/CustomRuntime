#pragma once
#include <vector>
#include <map>
#include <optional>
#include <string>

namespace types {

	template <typename T>
	struct ArrayND {
		std::vector<int> shape;
		std::vector<int> stride;
		std::vector<T> data;
	};

	enum LayerType {
		LAYER_CONV,
		LAYER_TRANSPOSE,
		LAYER_RESHAPE,
		LAYER_MAXPOOL,
		LAYER_SOFTMAX,
		LAYER_MUL,
		LAYER_ADD,
		LAYER_MATMUL,
		LAYER_RELU
	};

	inline const std::map<std::string, LayerType> StringToLayerType = {
		{"Reshape", LayerType::LAYER_RESHAPE},
		{"MaxPool", LayerType::LAYER_MAXPOOL},
		{"Softmax", LayerType::LAYER_SOFTMAX},
		{"Mul", LayerType::LAYER_MUL},
		{"Add", LayerType::LAYER_ADD},
		{"Transpose", LayerType::LAYER_TRANSPOSE},
		{"MatMul", LayerType::LAYER_MATMUL},
		{"Conv", LayerType::LAYER_CONV},
		{"Relu", LayerType::LAYER_RELU},
	};

	enum AttributeType{
		ATTR_STRIDES,
		ATTR_DILATIONS,
		ATTR_GROUP,
		ATTR_KERNEL_SHAPE,
		ATTR_PADS,
		ATTR_PERM,
	};

	inline const std::map<std::string, AttributeType> StringToAttributeType = {
		{"dilations", AttributeType::ATTR_DILATIONS},
		{"perm", AttributeType::ATTR_PERM},
		{"strides", AttributeType::ATTR_STRIDES},
		{"kernel_shape", AttributeType::ATTR_KERNEL_SHAPE},
		{"pads", AttributeType::ATTR_PADS},
		{"group", AttributeType::ATTR_GROUP},
	};

	struct LayerData {
		LayerType type;
		std::optional<ArrayND<double>> weights;
		std::optional<ArrayND<double>> bias;
		std::map<AttributeType, std::vector<int>> attributes;
	};

	template <typename T>
	void printVector(std::ostream& os, const std::vector<T>& vec);

	void printVectorDouble(std::ostream& os, const std::vector<double>& vec);

	std::ostream& operator<<(std::ostream& os, const LayerData& layerData);
	std::ostream& operator<<(std::ostream& os, const LayerType& type);
	std::ostream& operator<<(std::ostream& os, const AttributeType& type);

	
	template <typename T>
	std::ostream& operator<<(std::ostream& os, const ArrayND<T>& array);
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec);