#include "data_handler.h"
#include <fstream>
#include <stdexcept>
#include <filesystem>

namespace loader {
	DataHandler::DataHandler(const std::string& dataPath, const std::string& labelPath) {
		std::ifstream datasetStream(dataPath, std::ios::binary);

		if (!datasetStream.is_open()) {
			throw std::runtime_error("Failed to open data file: " + dataPath);
		}

		size_t file_size = std::filesystem::file_size(dataPath);

		if (file_size != 30730000) {
			throw std::runtime_error("Invalid data file size, recorded size: " + file_size);
		}

		std::vector<uint8_t> data(file_size);

		if (!datasetStream.read(reinterpret_cast<char*>(data.data()), file_size)) {
			throw std::runtime_error("Failed to read data file: " + dataPath);
		}

		datasetStream.close();

		this->datax.reserve(30720000);
		this->datay.reserve(10000);

		for (size_t i = 0; i < data.size(); i += 3073) {

			datay.push_back(data[i]);

			for (size_t j = 1; j < 3073; j++) {
				datax.push_back(data[i + j]);
			}
		}

		this->shape = { 10000, 3, 32, 32};
		this->stride = { 3072, 1024, 32, 1 };

		std::ifstream labelStream(labelPath);

		if (!labelStream.is_open()) {
			throw std::runtime_error("Failed to open label file: " + labelPath);
		}

		// reading the text file line by line and putting the labels in the vector

		for (std::string line; std::getline(labelStream, line);) {
			this->labels.push_back(line);
		}

		labelStream.close();
	}

	cv::Mat DataHandler::getImage(int index) {
		cv::Mat image(32, 32, CV_8UC3);
		for (int channel = 0; channel < 3; channel++) {
			for (int row = 0; row < 32; row++) {
				for (int col = 0; col < 32; col++) {
					// Access pixel at (row, col) in channel 'channel'
					image.at<cv::Vec3b>(row, col)[channel] = datax[index * 3072 + row * 32 + col + channel * 1024];
				}
			}
		}
		return image;
	}

	std::vector<cv::Mat> DataHandler::getImages() {
		std::vector<cv::Mat> images;
		for (int i = 0; i < 10000; i++) {
			images.push_back(getImage(i));
		}
		return images;
	}

}