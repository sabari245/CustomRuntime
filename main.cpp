// CustomRuntime.cpp : Defines the entry point for the application.
//

#include <fmt/format.h>
#include <fmt/core.h>

#include <opencv2/opencv.hpp>

#include "dataset_reader.h"
#include "model_reader.h"

#include "layers.h"

using namespace fmt::literals;

int main()
{

	reader::ModelReader modelReader(
		"C:/Users/sabar/source/repos/sabari245/CustomRuntime/model_color_export/data.json",
		"C:/Users/sabar/source/repos/sabari245/CustomRuntime/model_color_export/sequence.json"
	);

	auto layerData = modelReader.getLayerDataFromIndex(1);

	std::cout << layerData << std::endl;

	reader::DatasetReader datasetReader(
		"C:/Users/sabar/source/repos/sabari245/CustomRuntime/dataset/test_batch.bin",
		"C:/Users/sabar/source/repos/sabari245/CustomRuntime/dataset/batches.meta.txt"
	);


	std::cout << datasetReader.data << std::endl;

	layers::Layers engine;
	
	for (int i = 0; i < modelReader.getLayerCount(); i++) {
		auto layerData = modelReader.getLayerDataFromIndex(i);

		switch (layerData.type)
		{
		case types::LAYER_TRANSPOSE:
			break;
		case types::LAYER_CONV:
			break;
		case types::LAYER_RELU:
			break;
		case types::LAYER_MAXPOOL:
			break;
		case types::LAYER_RESHAPE:
			break;
		case types::LAYER_MATMUL:
			break;
		case types::LAYER_ADD:
			break;
		case types::LAYER_MUL:
			break;
		case types::LAYER_SOFTMAX:
			break;
		default:
			std::cout << "Unknown layer type: " << layerData.type << std::endl;
			return -1;
		}
	}

	return 0;
}
