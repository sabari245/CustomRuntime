# Create reader library
add_library(reader
    src/model_reader.cpp
    src/dataset_reader.cpp
)

# Add include directories
target_include_directories(reader
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Link dependencies
target_link_libraries(reader
    PUBLIC
        nlohmann_json::nlohmann_json
        # fmt::fmt
        utils
)

# Set C++ standard for reader library
set_target_properties(reader PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

if (CMAKE_VERSION VERSION_GREATER 3.12)
    set_target_properties(reader PROPERTIES
        CXX_STANDARD 20
    )
endif()

# Changes needed in root CMakeLists.txt:
# 1. Add 'add_subdirectory(reader)' after utils subdirectory
# 2. Replace direct source inclusion of reader files with library linkage:
#    - Remove "reader/src/model_reader.cpp" and "reader/src/dataset_reader.cpp" from add_executable() calls
#    - Remove reader include directory specifications
#    - Add 'target_link_libraries(<target> PRIVATE reader)' for CustomRuntime and UtilityTest targets
