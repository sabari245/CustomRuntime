#pragma once

#include "nlohmann/json.hpp"
#include <vector>
#include <string>
#include <optional>

namespace utils {
	
	enum ModelDataType {
		FLOAT32 = 1,
		INT64 = 7,
	};

	struct ModelData {
		std::string name;
		std::vector<int> shape;
		std::vector<int> stride;
		ModelDataType type;
		std::vector<double> data;
	};

	enum LayerType {
		TRANSPOSE,
		CONV2D,
		MUL,
		ADD,
		RELU,
		MAXPOOl,
		RESHAPE,
		MATMUL,
		SOFTMAX,
	};

	struct LayerData {
		LayerType type;
		std::optional<ModelData> weights;
		std::optional<ModelData> bias;
		std::optional<std::vector<int>> permutation;
		std::optional<std::vector<int>> padding;
		std::optional<std::vector<int>> stride;
		std::optional<std::vector<int>> kernel_shape;
		std::optional<std::vector<int>> shape;
	};

	class Model {
	private:
		nlohmann::ordered_json modelData;
		nlohmann::ordered_json modelSequence;

	public:
		Model(const std::string& modelDatafile, const std::string& modelSequenceFile);

		std::optional<ModelData> getModelDataFromName(const std::string& name);

		LayerData getLayerDataFromIndex(int index);

		int getLayerCount();
	};
}