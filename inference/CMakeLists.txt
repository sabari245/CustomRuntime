add_library(inference
    src/layers.cpp
)

target_include_directories(inference
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(inference
    PUBLIC
        utils
)

set_target_properties(inference PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

if (CMAKE_VERSION VERSION_GREATER 3.12)
    set_target_properties(inference PROPERTIES
        CXX_STANDARD 20
    )
endif()