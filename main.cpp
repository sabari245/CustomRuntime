// CustomRuntime.cpp : Defines the entry point for the application.
//

#include "main.h"
#include "model.h"
#include "format.h"
#include "data_handler.h"

#include "layers.h"

#include <fmt/format.h>
#include <fmt/core.h>

#include <opencv2/opencv.hpp>

using namespace fmt::literals;

int main()
{


	//utils::Model model(
	//	"C:/Users/sabar/source/repos/sabari245/CustomRuntime/model_color_new_export/data.json",
	//	"C:/Users/sabar/source/repos/sabari245/CustomRuntime/model_color_new_export/sequence.json"
	//);

	//for (int i = 0; i < model.getLayerCount(); i++) {
	//	std::cout << model.getLayerDataFromIndex(i) << std::endl;
	//}


	loader::DataHandler dataHandler(
		"C:/Users/sabar/source/repos/sabari245/CustomRuntime/dataset/test_batch.bin",
		"C:/Users/sabar/source/repos/sabari245/CustomRuntime/dataset/batches.meta.txt"
	);

	std::vector<int> first_image;

	first_image.reserve(32 * 32 * 3);

	for (int i = 0; i < 32 * 32 * 3; i++) {
		first_image.push_back(static_cast<int>(dataHandler.datax[i]));
	}

	const std::vector<int> shape = { 3, 32, 32 };
	const std::vector<int> stride = { 32 * 32, 32, 1 };

	const std::vector<int> permutation = { 1, 2, 0 };

	std::vector<int> outShape;
	std::vector<int> outStride;
	std::vector<int> first_image_transposed;

	layers::Layers instance;

	instance.transpose(
		shape,
		stride,
		first_image,
		permutation,
		outShape,
		outStride,
		first_image_transposed
	);

	// rendering the red image
	cv::Mat red_image(32, 32, CV_8UC1);

	for (int row = 0; row < 32; row++) {
		for (int col = 0; col < 32; col++) {
			red_image.at<uchar>(row, col) = static_cast<uchar>(first_image[row * 32 + col]);
		}
	}

	cv::namedWindow("Red Image", cv::WINDOW_NORMAL);

	cv::resizeWindow("Red Image", 600, 600);

	cv::imshow("Red Image", red_image);

	cv::waitKey(0);

	cv::Mat colored_image(32, 32, CV_8UC3);

	for (int row = 0; row < 32; row++) {
		for (int col = 0; col < 32; col++) {
			for (int channel = 0; channel < 3; channel++) {
				colored_image.at<cv::Vec3b>(row, col)[channel] = static_cast<uchar>(first_image_transposed[row * 32 * 3 + col * 3 + channel]);
			}
		}
	}

	cv::namedWindow("Colored Image", cv::WINDOW_NORMAL);
	
	cv::resizeWindow("Colored Image", 600, 600);

	cv::imshow("Colored Image", colored_image);

	cv::waitKey(0);

	std::cout << "Hello CMake." << std::endl;
	return 0;
}
