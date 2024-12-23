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

find_package(OpenCV REQUIRED)

# Add source to this project's executable.
add_executable (CustomRuntime "main.cpp" "utils/src/model.cpp" "utils/src/format.cpp" "utils/src/data_handler.cpp" "inference/src/layers.cpp")

target_include_directories(CustomRuntime PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/utils/include)
target_include_directories(CustomRuntime PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/inference/include)

target_link_libraries(CustomRuntime PRIVATE nlohmann_json::nlohmann_json)
target_link_libraries(CustomRuntime PRIVATE fmt::fmt)
target_link_libraries(CustomRuntime PRIVATE ${OpenCV_LIBS})

set(CMAKE_CXX_STANDARD 17)

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET CustomRuntime PROPERTY CXX_STANDARD 20)
endif()

# TODO: Add tests and install targets if needed.
