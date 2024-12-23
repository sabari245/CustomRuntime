#pragma once

#include <vector>
#include <string>

namespace loader {

	class DataHandler {
	public:
		std::vector<int> shape; // shape of the datax
		std::vector<int> stride; // stride of the datax for easier access
		std::vector<uint8_t> datax;

		std::vector<uint8_t> datay;

		std::vector<std::string> labels; // list of all the labels in the corresponding index

		DataHandler(const std::string& dataPath, const std::string& labelPath);


	};
}