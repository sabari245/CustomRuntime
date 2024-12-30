# Create utils library
add_library(utils
    src/types.cpp
    src/functions.cpp
)

# Add include directories
target_include_directories(utils
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Link dependencies
target_link_libraries(utils
    PUBLIC
        nlohmann_json::nlohmann_json
        # fmt::fmt
)

# Set C++ standard for utils library
set_target_properties(utils PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

if (CMAKE_VERSION VERSION_GREATER 3.12)
    set_target_properties(utils PROPERTIES
        CXX_STANDARD 20
    )
endif()