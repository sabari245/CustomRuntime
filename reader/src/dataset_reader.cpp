#include "dataset_reader.h"

#include <fstream>
#include <filesystem>
#include <iostream>

namespace reader {
	DatasetReader::DatasetReader(
		const std::string& dataPath,
		const std::string& labelPath
	) {
		this->readData(dataPath);

		std::cout << "[INFO] Successfully read the data file" << std::endl;

		this->readLabels(labelPath);

		std::cout << "[INFO] Successfully read the label file" << std::endl;
	}

	void DatasetReader::readData(const std::string& dataPath) {
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

		this->data.shape = { 10000, 3, 32, 32 };
		this->data.stride = { 3072, 1024, 32, 1 };

		this->data.data.reserve(30720000);
		this->labels.reserve(10000);

		for (size_t i = 0; i < data.size(); i += 3073) {
			this->labels.push_back(data[i]);
			for (size_t j = 1; j < 3073; j++) {
				this->data.data.push_back(data[i + j]);
			}
		}
	}

	void DatasetReader::readLabels(const std::string& labelPath) {
		std::ifstream labelStream(labelPath);
		if (!labelStream.is_open()) {
			throw std::runtime_error("Failed to open label file: " + labelPath);
		}
		// reading the text file line by line and putting the labels in the vector
		for (std::string line; std::getline(labelStream, line);) {
			this->labelNames.push_back(line);
		}
		labelStream.close();
	}
}