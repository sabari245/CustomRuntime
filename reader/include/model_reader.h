#pragma once

#include "types.h"
#include "nlohmann/json.hpp"
#include <map>

namespace reader
{
	class ModelReader
	{
	private:
		std::map<std::string, types::ArrayND<double>> nameModelDataMap;
		std::vector<types::LayerData> layerDataList;

		nlohmann::ordered_json getDataFromFile(const std::string &filename);

		void parseDataFile(const nlohmann::ordered_json &modelData);
		void parseSequenceFile(const nlohmann::ordered_json &modelSequence);

	public:
		std::vector<int> getStrideFromShape(const std::vector<int> &shape);
		ModelReader(const std::string &modelDatafile, const std::string &modelSequenceFile);
		types::LayerData getLayerDataFromIndex(int index);

		types::LayerData &getLayerDataRefFromIndex(int index);
		int getLayerCount();
	};
}