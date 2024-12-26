# CMakeList.txt : CMake project for CustomRuntime, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project ("CustomRuntime")

include(FetchContent)

FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
FetchContent_MakeAvailable(json)

FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt
  GIT_TAG        e69e5f977d458f2650bb346dadf2ad30c5320281) # 10.2.1
FetchContent_MakeAvailable(fmt)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

find_package(OpenCV REQUIRED)

enable_testing()
include(GoogleTest)

# Add source to this project's executable.
add_executable (CustomRuntime "main.cpp"    "inference/src/layers.cpp" "utils/src/types.cpp" "reader/src/model_reader.cpp" "reader/src/dataset_reader.cpp" "utils/src/functions.cpp")

target_include_directories(CustomRuntime PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/utils/include)
target_include_directories(CustomRuntime PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/inference/include)
target_include_directories(CustomRuntime PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/reader/include)

target_link_libraries(CustomRuntime PRIVATE nlohmann_json::nlohmann_json)
target_link_libraries(CustomRuntime PRIVATE fmt::fmt)
target_link_libraries(CustomRuntime PRIVATE ${OpenCV_LIBS})
target_link_libraries(CustomRuntime PRIVATE GTest::gtest_main)

add_executable(UtilityTest "tests/utility_test.cpp" "utils/src/types.cpp" "utils/src/functions.cpp" "reader/src/model_reader.cpp" "reader/src/dataset_reader.cpp")

target_include_directories(UtilityTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/utils/include)
target_include_directories(UtilityTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/reader/include)

target_link_libraries(UtilityTest PRIVATE nlohmann_json::nlohmann_json)
target_link_libraries(UtilityTest PRIVATE fmt::fmt)
target_link_libraries(UtilityTest PRIVATE GTest::gtest_main)

add_executable(LayerTest "tests/layer_test.cpp" "utils/src/types.cpp" "utils/src/functions.cpp" "inference/src/layers.cpp")

target_include_directories(LayerTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/utils/include)
target_include_directories(LayerTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/inference/include)

target_link_libraries(LayerTest PRIVATE nlohmann_json::nlohmann_json)
target_link_libraries(LayerTest PRIVATE fmt::fmt)
target_link_libraries(LayerTest PRIVATE GTest::gtest_main)

gtest_discover_tests(UtilityTest)
gtest_discover_tests(LayerTest)

# Setting the C++ Standard globally for the project
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON) # Make sure C++17 is required
set(CMAKE_CXX_EXTENSIONS OFF)      # Disallow extensions

# Apply the C++ standard to ALL targets. This is a better way to set it.
set_property(TARGET CustomRuntime PROPERTY CXX_STANDARD 17)
set_property(TARGET CustomRuntime PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET CustomRuntime PROPERTY CXX_EXTENSIONS OFF)
set_property(TARGET UtilityTest PROPERTY CXX_STANDARD 17)
set_property(TARGET UtilityTest PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET UtilityTest PROPERTY CXX_EXTENSIONS OFF)
set_property(TARGET LayerTest PROPERTY CXX_STANDARD 17)
set_property(TARGET LayerTest PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET LayerTest PROPERTY CXX_EXTENSIONS OFF)

if (CMAKE_VERSION VERSION_GREATER 3.12)
    set_property(TARGET CustomRuntime PROPERTY CXX_STANDARD 20)
    set_property(TARGET UtilityTest PROPERTY CXX_STANDARD 20)
    set_property(TARGET LayerTest PROPERTY CXX_STANDARD 20)
endif()

# TODO: Add tests and install targets if needed.
