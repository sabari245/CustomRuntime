#include "types.h"
#include <iostream>
#include <iomanip>

namespace types {

    // Helper function to print vector with truncation for large vectors
    template <typename T>
    void printVector(std::ostream& os, const std::vector<T>& vec) {
        if (vec.size() > 6) {
            for (size_t i = 0; i < 3; ++i) {
                os << vec[i] << " ";
            }
            os << "... ";
            for (size_t i = vec.size() - 3; i < vec.size(); ++i) {
                os << vec[i] << " ";
            }
        }
        else {
            for (const auto& val : vec) {
                os << val << " ";
            }
        }
    }

    template <>
    void printVector(std::ostream& os, const std::vector<uint8_t>& vec) {
        if (vec.size() > 6) {
            for (size_t i = 0; i < 3; ++i) {
                os << static_cast<int>(vec[i]) << " ";
            }
            os << "... ";
            for (size_t i = vec.size() - 3; i < vec.size(); ++i) {
                os << static_cast<int>(vec[i]) << " ";
            }
        }
        else {
            for (const auto& val : vec) {
                os << static_cast<int>(val) << " ";
            }
        }
    }


    // Specialization for double to ensure full precision
    void printVectorDouble(std::ostream& os, const std::vector<double>& vec) {
        if (vec.size() > 6) {
            os << std::fixed << std::setprecision(16);
            for (size_t i = 0; i < 3; ++i) {
                os << vec[i] << " ";
            }
            os << "... ";
            for (size_t i = vec.size() - 3; i < vec.size(); ++i) {
                os << vec[i] << " ";
            }
        }
        else {
            os << std::fixed << std::setprecision(16);
            for (const auto& val : vec) {
                os << val << " ";
            }
        }
    }

    std::ostream& operator<<(std::ostream& os, const LayerType& type) {
        switch (type) {
        case LAYER_CONV: os << "Conv"; break;
        case LAYER_TRANSPOSE: os << "Transpose"; break;
        case LAYER_RESHAPE: os << "Reshape"; break;
        case LAYER_MAXPOOL: os << "MaxPool"; break;
        case LAYER_SOFTMAX: os << "Softmax"; break;
        case LAYER_MUL: os << "Mul"; break;
        case LAYER_ADD: os << "Add"; break;
        case LAYER_MATMUL: os << "MatMul"; break;
        case LAYER_RELU: os << "Relu"; break;
        }
        return os;
    }

    std::ostream& operator<<(std::ostream& os, const AttributeType& type) {
        switch (type) {
        case ATTR_STRIDES: os << "strides"; break;
        case ATTR_DILATIONS: os << "dilations"; break;
        case ATTR_GROUP: os << "group"; break;
        case ATTR_KERNEL_SHAPE: os << "kernel_shape"; break;
        case ATTR_PADS: os << "pads"; break;
        case ATTR_PERM: os << "perm"; break;
        }
        return os;
    }

    std::ostream& operator<<(std::ostream& os, const LayerData& layerData) {
        os << "LayerType: " << layerData.type << "\n";
        if (layerData.weights) {
            os << "Weights: " << *layerData.weights << "\n";
        }
        if (layerData.bias) {
            os << "Bias: " << *layerData.bias << "\n";
        }
        os << "Attributes: \n";
        for (const auto& [key, value] : layerData.attributes) {
            os << "  " << key << ": ";
            printVector(os, value);
            os << "\n";
        }
        return os;
    }

    template <typename T>
    std::ostream& operator<<(std::ostream& os, const ArrayND<T>& array) {
        os << "Shape: ";
        printVector(os, array.shape);
        os << "\nStride: ";
        printVector(os, array.stride);
        os << "\nData: ";
        printVector(os, array.data);
        return os;
    }

    // Explicit template instantiations
    template std::ostream& operator<<(std::ostream& os, const ArrayND<int>& array);
    template std::ostream& operator<<(std::ostream& os, const ArrayND<double>& array);
    template std::ostream& operator<<(std::ostream& os, const ArrayND<uint8_t>& array);
}


template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec) {
    types::printVector(os, vec);
    return os;
}

template <>
std::ostream& operator<<(std::ostream& os, const std::vector<uint8_t>& vec) {
    types::printVector(os, vec);
    return os;
}

template <>
std::ostream& operator<<(std::ostream& os, const std::vector<double>& vec) {
    types::printVectorDouble(os, vec);
    return os;
}

template std::ostream& operator<<(std::ostream& os, const std::vector<int>& vec);
template std::ostream& operator<<(std::ostream& os, const std::vector<double>& vec);
template std::ostream& operator<<(std::ostream& os, const std::vector<uint8_t>& vec);