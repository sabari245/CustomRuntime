#pragma once

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

namespace loader {

	class DataHandler {
	public:
		std::vector<int> shape; // shape of the datax
		std::vector<int> stride; // stride of the datax for easier access
		std::vector<uint8_t> datax;

		std::vector<uint8_t> datay;

		std::vector<std::string> labels; // list of all the labels in the corresponding index

		DataHandler(const std::string& dataPath, const std::string& labelPath);

		cv::Mat getImage(int index);

		std::vector<cv::Mat> getImages();
	};
}