#include "model.h"
#include <fstream>
#include <iostream>

namespace utils {
	Model::Model(const std::string& modelDatafile, const std::string& modelSequenceFile) {
		std::ifstream modelDataStream(modelDatafile);

		if (!modelDataStream.is_open()) {
			std::cerr << "Failed to open model data file: " << modelDatafile << std::endl;
			exit(1);
		}

		modelDataStream >> modelData;
		modelDataStream.close();
		std::ifstream modelSequenceStream(modelSequenceFile);

		if (!modelSequenceStream.is_open()) {
			std::cerr << "Failed to open model sequence file: " << modelSequenceFile << std::endl;
			exit(1);
		}

		modelSequenceStream >> modelSequence;
		modelSequenceStream.close();
	}

	std::optional<ModelData> Model::getModelDataFromName(const std::string& name) {
		
		ModelData result;

		for (auto& model : modelData) {
			if (model["name"] == name) {
				result.data = model["data"].get<std::vector<double>>();
				result.name = model["name"].get<std::string>();
				result.shape = model["shape"].get<std::vector<int>>();
				result.stride = model["stride"].get<std::vector<int>>();
				result.type = static_cast<ModelDataType>(model["data_type"].get<int>());

				return result;
			}
		}

		return std::nullopt;
	}

	LayerData Model::getLayerDataFromIndex(int index) {
		LayerData layerData;

		if (index >= modelSequence.size()) {
			std::cerr << "Index out of bounds" << std::endl;
			exit(1);
		}
		
		auto layer = modelSequence[index];

		if (layer["operation"] == "Transpose") {
			layerData.type = LayerType::TRANSPOSE;
			layerData.permutation = layer["attributes"]["perm"].get<std::vector<int>>();
		}
		else if (layer["operation"] == "Conv") {
			layerData.type = LayerType::CONV2D;

			std::vector<std::string> inputs = layer["inputs"].get<std::vector<std::string>>();

			layerData.weights = this->getModelDataFromName(inputs[1]);
			layerData.bias = this->getModelDataFromName(inputs[2]);

			if (layer["attributes"].contains("pads")) {
				layerData.padding = layer["attributes"]["pads"].get<std::vector<int>>();
			}
			else {
				layerData.padding = { 0, 0, 0, 0 };
			}

			layerData.stride = layer["attributes"]["strides"].get<std::vector<int>>();
			layerData.kernel_shape = layer["attributes"]["kernel_shape"].get<std::vector<int>>();
		}
		else if (layer["operation"] == "Mul") {
			layerData.type = LayerType::MUL;
			layerData.weights = this->getModelDataFromName(layer["inputs"][1].get<std::string>());
		}
		else if (layer["operation"] == "Add") {
			layerData.type = LayerType::ADD;
			layerData.weights = this->getModelDataFromName(layer["inputs"][1].get<std::string>());
		}
		else if (layer["operation"] == "Relu") {
			layerData.type = LayerType::RELU;
		}
		else if (layer["operation"] == "MaxPool") {
			layerData.type = LayerType::MAXPOOl;
			layerData.stride = layer["attributes"]["strides"].get<std::vector<int>>();
			layerData.kernel_shape = layer["attributes"]["kernel_shape"].get<std::vector<int>>();
		}
		else if (layer["operation"] == "Reshape") {
			layerData.type = LayerType::RESHAPE;

			if (layer["attributes"].contains("shape"))
				layerData.shape = layer["attributes"]["shape"].get<std::vector<int>>();

		}
		else if (layer["operation"] == "MatMul") {
			layerData.type = LayerType::MATMUL;
			layerData.weights = this->getModelDataFromName(layer["inputs"][1].get<std::string>());
		}
		else if (layer["operation"] == "Softmax") {
			layerData.type = LayerType::SOFTMAX;
		}
		else {
			std::cerr << "Invalid operation" << std::endl;
			exit(1);
		}

		return layerData;
	}

	int Model::getLayerCount() {
		return modelSequence.size();
	}
}