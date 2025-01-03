## CustomRuntime - A C++ Project with CMake

This project contains a custom runtime system built using C++. It implements a custom datatype for working with multidimensional array. It utilizes several external libraries to achieve functionalities like JSON parsing, formatting, and unit testing.

### Prerequisites

* CMake (version 3.8 or later): [https://cmake.org/](https://cmake.org/)
* A C++ compiler (tested with GCC and MSVC)
* External Libraries:
    * nlohmann/json ([https://github.com/nlohmann/json](https://github.com/nlohmann/json))
    * Google Test ([https://github.com/google/googletest](https://github.com/google/googletest))
    * ~fmtlib/fmt ([https://github.com/fmtlib/fmt](https://github.com/fmtlib/fmt))~
    * ~OpenCV ([https://opencv.org/](https://opencv.org/))~

> **Note:** OpenCV need to be installed on your system. Make sure to set the path variables correctly if you are using Windows.

### Building the Project

1. Clone the repository or download the source code.
2. Create a build directory (e.g., `out`).
3. Navigate to the build directory.
4. Run the following commands:

```bash
mkdir out && cd out
cmake ..

cmake --build . # if you are on windoww
make # if you are on linux
```

This will generate the executable `CustomRuntime` and unit test executables (`UtilityTest` and `LayerTest`) in the build directory.

### Running the Program

You can run the `CustomRuntime` executable directly:

```bash
./CustomRuntime
```

The behavior of the program depends on the specific implementation in the source code (not provided in this example).

### Running Unit Tests

To run the unit tests, navigate to the build directory and execute:

```bash
./UtilityTest
./LayerTest
```

These commands will run the tests and report the results.

### Project Structure

The project is organized with the following directories:

* `inference`: Contains the custom implementations of the layers in the model like `convolution`, `matrixmultiplication`, `transpose` and so on.
* `reader`: Contains the logic for reading the custom model files and input data.
* `tests`: Contains unit test source code.
* `utils`: Contains the definition of the custom datatype which is used throughout the project to hold the Multi Dimensional Array `ArrayND`.

### Reader Details

The reader module has a namespace `reader` which contains the classes `ModelReader` and `DataReader`. 

* The `ModelReader` class reads the model file and creates a model object. It uses nlohmann/json library to parse the JSON file. The model object is split into two json files, one to store the weights and biases and another one to store the layer and sequence information.
* The `DataReader` class reads the input data file and creates an input data object. The `CIFAR-10` dataset is used as the input data. The dataset is downloaded as a binary file from the [Official CIFAR-10 website](https://www.cs.toronto.edu/~kriz/cifar.html).

> Download the dataset using the following link [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz)

> After placing the models and dataset, make sure to update the paths in the `main.cpp` file.

### Utility Details

This module contains two namespaces `types` and `utility`. 

* The types contains the definition of the custom datatypes including `ArrayND` and implements overload for ostream operators.
* The Utility contains functions to perform various operations on the `ArrayND` datatype like `create`, `transpose`, `reshape`, `flatten`, `slice` and so on.

### Inference Details

This module contains the definition of various layers all present within the class `layers::Layer`.
