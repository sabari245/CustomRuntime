// CustomRuntime.cpp : Defines the entry point for the application.
//

#include <fmt/format.h>
#include <fmt/core.h>

#include <opencv2/opencv.hpp>

//#include "model_reader.h"
//#include "dataset_reader.h"

#include "reader/include/dataset_reader.h"
#include "reader/include/model_reader.h"

using namespace fmt::literals;

int main()
{

	reader::ModelReader modelReader(
		"C:/Users/sabar/source/repos/sabari245/CustomRuntime/model_color_export/data.json",
		"C:/Users/sabar/source/repos/sabari245/CustomRuntime/model_color_export/sequence.json"
	);

	auto layerData = modelReader.getLayerDataFromIndex(1);

	std::cout << layerData << std::endl;

	//reader::DatasetReader datasetReader(
	//	"C:/Users/sabar/source/repos/sabari245/CustomRuntime/dataset/test_batch.bin",
	//	"C:/Users/sabar/source/repos/sabari245/CustomRuntime/dataset/batches.meta.txt"
	//);

	//std::cout << datasetReader.data << std::endl;

	return 0;
}
