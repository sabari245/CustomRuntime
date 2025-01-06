#pragma once

#include <string>
#include <vector>

#include "types.h"

namespace reader
{
	class DatasetReader
	{
	private:
		void readData(const std::string &dataPath);
		void readLabels(const std::string &labelPath);

	public:
		types::ArrayND<uint8_t> data;
		std::vector<uint8_t> labels;

		std::vector<std::string> labelNames;

		DatasetReader(
			const std::string &dataPath,
			const std::string &labelPath);
	};
}