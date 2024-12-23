#pragma once

#include "model.h"
#include <iostream>
#include <vector>

// Forward declarations for operator overloads
std::ostream& operator<<(std::ostream& os, std::vector<int> vec);
std::ostream& operator<<(std::ostream& os, const std::vector<double>& vec);
std::ostream& operator<<(std::ostream& os, const utils::ModelDataType& type);
std::ostream& operator<<(std::ostream& os, const utils::ModelData& modelData);
std::ostream& operator<<(std::ostream& os, const utils::LayerType& type);
std::ostream& operator<<(std::ostream& os, const utils::LayerData& layerData);