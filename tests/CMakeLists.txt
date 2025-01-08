# Tests CMakeLists.txt

# Include necessary dependencies
include(FetchContent)
FetchContent_Declare(
  googletest
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Enable CTest
include(CTest)
enable_testing()
include(GoogleTest)

# Utility Test
add_executable(UtilityTest "utility_test.cpp")

target_link_libraries(UtilityTest PRIVATE 
    GTest::gtest_main
    utils
)

# Layer Test
add_executable(LayerTest "layer_test.cpp")

target_link_libraries(LayerTest PRIVATE 
    GTest::gtest_main
    utils
    inference
    fftw3
)

# Set C++ Standard for test targets
set_property(TARGET UtilityTest PROPERTY CXX_STANDARD 17)
set_property(TARGET UtilityTest PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET UtilityTest PROPERTY CXX_EXTENSIONS OFF)

set_property(TARGET LayerTest PROPERTY CXX_STANDARD 17)
set_property(TARGET LayerTest PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET LayerTest PROPERTY CXX_EXTENSIONS OFF)

if(CMAKE_VERSION VERSION_GREATER 3.12)
    set_property(TARGET UtilityTest PROPERTY CXX_STANDARD 20)
    set_property(TARGET LayerTest PROPERTY CXX_STANDARD 20)
endif()

# Add tests to CTest
add_test(NAME UtilityTest COMMAND UtilityTest)
add_test(NAME LayerTest COMMAND LayerTest)

# Discover tests with GTest
gtest_discover_tests(UtilityTest)
gtest_discover_tests(LayerTest)