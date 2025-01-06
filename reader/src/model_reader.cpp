#include "model_reader.h"
#include <fstream>
#include <iostream>

namespace reader
{
	ModelReader::ModelReader(const std::string &modelDatafile, const std::string &modelSequenceFile)
	{
		auto modelDataJson = this->getDataFromFile(modelDatafile);
		auto modelSequenceJson = this->getDataFromFile(modelSequenceFile);

		std::cout << "[INFO] Successfully read the model and sequence file" << std::endl;

		this->parseDataFile(modelDataJson);

		std::cout << "[INFO] Successfully parsed the model data" << std::endl;

		this->parseSequenceFile(modelSequenceJson);

		std::cout << "[INFO] Successfully parsed the model sequence" << std::endl;
	}

	nlohmann::ordered_json ModelReader::getDataFromFile(const std::string &filename)
	{
		std::ifstream file(filename);

		if (!file.is_open())
		{
			throw std::runtime_error("Could not open file: " + filename);
		}

		nlohmann::ordered_json data;
		file >> data;

		file.close();

		return data;
	}

	void ModelReader::parseDataFile(const nlohmann::ordered_json &modelData)
	{
		for (const auto &model : modelData)
		{
			types::ArrayND<double> array;
			array.data = model["data"].get<std::vector<double>>();
			array.shape = model["shape"].get<std::vector<int>>();
			array.stride = this->getStrideFromShape(array.shape);
			this->nameModelDataMap[model["name"]] = array;
		}
	}

	void ModelReader::parseSequenceFile(const nlohmann::ordered_json &modelSequence)
	{
		for (const auto &layer : modelSequence)
		{
			types::LayerData layerData;

			if (types::StringToLayerType.contains(layer["operation"]))
			{
				layerData.type = types::StringToLayerType.at(layer["operation"]);
			}
			else
			{
				throw std::runtime_error("Invalid layer type: " + layer["operation"].get<std::string>());
			}

			if (layer["inputs"].size() > 1)
			{
				layerData.weights = this->nameModelDataMap[layer["inputs"][1]];
			}

			if (layer["inputs"].size() > 2)
			{
				layerData.bias = this->nameModelDataMap[layer["inputs"][2]];
			}

			if (layer.contains("attributes"))
			{
				for (const auto &[key, value] : layer["attributes"].items())
				{
					if (types::StringToAttributeType.contains(key))
					{
						if (value.is_array())
						{
							layerData.attributes[types::StringToAttributeType.at(key)] = value.get<std::vector<int>>();
						}
						else if (value.is_number_integer())
						{
							layerData.attributes[types::StringToAttributeType.at(key)] = {value.get<int>()};
						}
						else
						{
							throw std::runtime_error("Invalid value type for attribute: " + key);
						}
					}
					else
					{
						throw std::runtime_error("Invalid attribute type: " + key);
					}
				}
			}

			this->layerDataList.push_back(layerData);
		}
	}

	std::vector<int> ModelReader::getStrideFromShape(const std::vector<int> &shape)
	{
		std::vector<int> stride(shape.size(), 1);
		for (int i = shape.size() - 2; i >= 0; i--)
		{
			stride[i] = shape[i + 1] * stride[i + 1];
		}
		return stride;
	}

	types::LayerData ModelReader::getLayerDataFromIndex(int index)
	{
		if (index >= this->layerDataList.size())
		{
			throw std::runtime_error("Index out of bounds");
		}
		return this->layerDataList[index];
	}

	types::LayerData &ModelReader::getLayerDataRefFromIndex(int index)
	{
		if (index >= this->layerDataList.size())
		{
			throw std::runtime_error("Index out of bounds");
		}
		return this->layerDataList[index];
	}

	int ModelReader::getLayerCount()
	{
		return this->layerDataList.size();
	}
}