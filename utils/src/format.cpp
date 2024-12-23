#include "format.h"
#include <fmt/format.h>
#include <iostream>
#include <iomanip>

std::ostream& operator<<(std::ostream& os, std::vector<int> vec) {
    os << "[";
    for (int i = 0; i < vec.size(); i++) {
        os << vec[i];
        if (i != vec.size() - 1) {
            os << ", ";
        }
    }
    os << "]";
    return os;
}

std::ostream& operator<<(std::ostream& os, const std::vector<uint8_t>& vec) {
    os << "[";
    size_t size = vec.size();

    if (size <= 6) {
        for (size_t i = 0; i < size; ++i) {
            os << static_cast<int>(vec[i]);
            if (i != size - 1) {
                os << ", ";
            }
        }
    }
    else {
        for (size_t i = 0; i < 3; ++i) {
            os << static_cast<int>(vec[i]);
            if (i != 2) {
                os << ", ";
            }
        }

        os << ", ..., ";
        for (size_t i = size - 3; i < size; ++i) {
            os << static_cast<int>(vec[i]);
            if (i != size - 1) {
                os << ", ";
            }
        }
    }
    os << "]";
    return os;
}

std::ostream& operator<<(std::ostream& os, uint8_t val) {
	os << static_cast<int>(val);
	return os;
}

std::ostream& operator<<(std::ostream& os, const std::vector<double>& vec) {
    os << "[";
    size_t size = vec.size();

    if (size <= 6) {
        for (size_t i = 0; i < size; ++i) {
            os << vec[i];
            if (i != size - 1) {
                os << ", ";
            }
        }
    }
    else {
        for (size_t i = 0; i < 3; ++i) {
            os << vec[i];
            if (i != 2) {
                os << ", ";
            }
        }

        os << ", ..., ";
        for (size_t i = size - 3; i < size; ++i) {
            os << vec[i];
            if (i != size - 1) {
                os << ", ";
            }
        }
    }
    os << "]";
    return os;
}

std::ostream& operator<<(std::ostream& os, const utils::ModelDataType& type) {
    switch (type) {
    case utils::ModelDataType::FLOAT32:
        os << "FLOAT32";
        break;
    case utils::ModelDataType::INT64:
        os << "INT64";
        break;
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, const utils::ModelData& modelData) {

    os << "Name: " << modelData.name << "\n";
    os << "Shape: " << modelData.shape << "\n";
    os << "Stride: " << modelData.stride << "\n";
    os << "Type: " << modelData.type << "\n";
    os << "Data: " << modelData.data << "\n";

    return os;
}

std::ostream& operator<<(std::ostream& os, const utils::LayerType& type) {
    switch (type) {
    case utils::LayerType::TRANSPOSE:
        os << "TRANSPOSE";
        break;
    case utils::LayerType::CONV2D:
        os << "CONV2D";
        break;
    case utils::LayerType::MUL:
        os << "MUL";
        break;
    case utils::LayerType::ADD:
        os << "ADD";
        break;
    case utils::LayerType::RELU:
        os << "RELU";
        break;
    case utils::LayerType::MAXPOOl:
        os << "MAXPOOL";
        break;
    case utils::LayerType::RESHAPE:
        os << "RESHAPE";
        break;
    case utils::LayerType::MATMUL:
        os << "MATMUL";
        break;
    case utils::LayerType::SOFTMAX:
        os << "SOFTMAX";
        break;
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, const utils::LayerData& layerData) {
    os << "Type: " << layerData.type << "\n";
    if (layerData.weights.has_value()) {
        os << "Weights: " << layerData.weights.value() << "\n";
    }
    if (layerData.bias.has_value()) {
        os << "Bias: " << layerData.bias.value() << "\n";
    }
    if (layerData.permutation.has_value()) {
        os << "Permuation: " << layerData.permutation.value() << "\n";
    }
    if (layerData.padding.has_value()) {
        os << "Padding: " << layerData.padding.value() << "\n";
    }
    if (layerData.stride.has_value()) {
        os << "Stride: " << layerData.stride.value() << "\n";
    }
    if (layerData.kernel_shape.has_value()) {
        os << "Kernel Shape: " << layerData.kernel_shape.value() << "\n";
    }
    if (layerData.shape.has_value()) {
        os << "Shape: " << layerData.shape.value() << "\n";
    }
    return os;
}