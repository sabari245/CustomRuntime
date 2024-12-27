#include <fmt/format.h>
#include <fmt/core.h>
#include <stdexcept>
#include <time.h>
#include <numeric>
#include <vector>
#include <algorithm>

#include <opencv2/opencv.hpp>

#include "dataset_reader.h"
#include "model_reader.h"
#include "layers.h"
#include "types.h"

using namespace fmt::literals;

struct ModelMetrics {
    double avg_inference_time;
    double throughput;
    double accuracy;
};

// Function to run inference on a single input
std::pair<types::ArrayND<double>, double> runInference(
    const types::ArrayND<double>& input,
    layers::Layers& engine,
    reader::ModelReader& modelReader) {

    clock_t start = clock();
    auto current_input = input;

    for (int i = 0; i < modelReader.getLayerCount(); i++) {
        auto layerData = modelReader.getLayerDataFromIndex(i);

        switch (layerData.type) {
        case types::LAYER_TRANSPOSE:
            current_input = engine.transpose(
                current_input,
                layerData.attributes.at(types::ATTR_PERM)
            );
            break;

        case types::LAYER_CONV: {
            std::vector<int> padding;
            try {
                padding = layerData.attributes.at(types::ATTR_PADS);
            }
            catch (const std::exception& e) {
                padding = { 0, 0, 0, 0 };
            }
            current_input = engine.conv2d(
                current_input,
                layerData.weights.value(),
                layerData.bias.value(),
                padding,
                layerData.attributes.at(types::ATTR_STRIDES),
                layerData.attributes.at(types::ATTR_KERNEL_SHAPE)
            );
            break;
        }

        case types::LAYER_RELU:
            current_input = engine.Relu(current_input);
            break;

        case types::LAYER_MAXPOOL:
            current_input = engine.maxPool(
                current_input,
                layerData.attributes.at(types::ATTR_KERNEL_SHAPE),
                layerData.attributes.at(types::ATTR_STRIDES)
            );
            break;

        case types::LAYER_RESHAPE:
            current_input = engine.Reshape(
                current_input,
                layerData.weights.value()
            );
            break;

        case types::LAYER_MATMUL:
            current_input = engine.MatMul(
                current_input,
                layerData.weights.value()
            );
            break;

        case types::LAYER_ADD: {
            types::ArrayND<double> bias = layerData.weights.value();
            if (bias.shape.size() == 1) {
                bias = utility::reshape(bias, { 1, bias.shape[0] });
            }
            current_input = engine.Add(current_input, bias);
            break;
        }

        case types::LAYER_MUL:
            current_input = engine.Mul(
                current_input,
                layerData.weights.value()
            );
            break;

        case types::LAYER_SOFTMAX:
            current_input = engine.Softmax(current_input);
            break;

        default:
            throw std::runtime_error("Unknown layer type: " + std::to_string(layerData.type));
        }
    }

    double inference_time = (double)(clock() - start) / CLOCKS_PER_SEC;
    return { current_input, inference_time };
}

// Function to get predicted class from output
int getPredictedClass(const types::ArrayND<double>& output) {
    return std::max_element(output.data.begin(), output.data.end()) - output.data.begin();
}

ModelMetrics evaluateModel(const std::string& model_path, const std::string& sequence_path,
    const std::string& dataset_path, const std::string& meta_path,
    int num_samples = 100) {

    reader::ModelReader modelReader(model_path, sequence_path);
    reader::DatasetReader datasetReader(dataset_path, meta_path);
    layers::Layers engine;

    std::vector<double> inference_times;
    int correct_predictions = 0;

    // Convert dataset to double
    types::ArrayND<double> datasetDouble = utility::convertUint8ToType<double>(datasetReader.data);

    for (int i = 0; i < num_samples; i++) {
        // Prepare input
        types::ArrayND<double> input = utility::slice(datasetDouble, i, i + 1);
        input = utility::transpose(input, { 0, 2, 3, 1 });

        // Run inference
        auto [output, inference_time] = runInference(input, engine, modelReader);
        inference_times.push_back(inference_time);

        // Check accuracy
        int predicted_class = getPredictedClass(output);
        if (predicted_class == static_cast<int>(datasetReader.labels[i])) {
            correct_predictions++;
        }
    }

    // Calculate metrics
    double avg_inference_time = std::accumulate(inference_times.begin(), inference_times.end(), 0.0) / num_samples;
    double throughput = num_samples / std::accumulate(inference_times.begin(), inference_times.end(), 0.0);
    double accuracy = static_cast<double>(correct_predictions) / num_samples;

    return { avg_inference_time, throughput, accuracy };
}

int main() {
    const std::string MODEL_PATH = "C:/Users/sabar/source/repos/sabari245/CustomRuntime/model_color_export/data.json";
    const std::string SEQUENCE_PATH = "C:/Users/sabar/source/repos/sabari245/CustomRuntime/model_color_export/sequence.json";
    const std::string DATASET_PATH = "C:/Users/sabar/source/repos/sabari245/CustomRuntime/dataset/test_batch.bin";
    const std::string META_PATH = "C:/Users/sabar/source/repos/sabari245/CustomRuntime/dataset/batches.meta.txt";

    try {
        ModelMetrics metrics = evaluateModel(MODEL_PATH, SEQUENCE_PATH, DATASET_PATH, META_PATH);

        std::cout << fmt::format("Model Evaluation Results (100 samples):\n"
            "Average Inference Time: {:.4f} seconds\n"
            "Throughput: {:.2f} inferences per second\n"
            "Accuracy: {:.2f}%\n",
            metrics.avg_inference_time,
            metrics.throughput,
            metrics.accuracy * 100);
    }
    catch (const std::exception& e) {
        std::cerr << "Error during model evaluation: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}