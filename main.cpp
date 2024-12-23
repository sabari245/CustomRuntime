// CustomRuntime.cpp : Defines the entry point for the application.
//

#include "main.h"
#include "model.h"
#include "format.h"

#include <fmt/format.h>
#include <fmt/core.h>

using namespace fmt::literals;

int main()
{


	utils::Model model(
		"C:/Users/sabar/source/repos/sabari245/CustomRuntime/model_color_new_export/data.json",
		"C:/Users/sabar/source/repos/sabari245/CustomRuntime/model_color_new_export/sequence.json"
	);

	for (int i = 0; i < model.getLayerCount(); i++) {
		std::cout << model.getLayerDataFromIndex(i) << std::endl;
	}


	std::cout << "Hello CMake." << std::endl;
	return 0;
}
