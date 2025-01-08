# CMakeList.txt : CMake project for CustomRuntime, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.11)

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project ("CustomRuntime")

include(FetchContent)

FetchContent_Declare(
  json 
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
)
FetchContent_MakeAvailable(json)


add_subdirectory("utils")
add_subdirectory("reader")
add_subdirectory("inference")
add_subdirectory("tests")
add_subdirectory("fftw-3.3.10")
find_package(OpenMP REQUIRED)


# Add source to this project's executable.
add_executable (CustomRuntime "main.cpp")

target_link_libraries(CustomRuntime PRIVATE nlohmann_json::nlohmann_json)
target_link_libraries(CustomRuntime PRIVATE utils)
target_link_libraries(CustomRuntime PRIVATE reader)
target_link_libraries(CustomRuntime PRIVATE inference)
target_link_libraries(CustomRuntime PRIVATE fftw3)
if(OpenMP_CXX_FOUND)
    target_link_libraries(CustomRuntime PRIVATE OpenMP::OpenMP_CXX)
endif()


# Setting the C++ Standard globally for the project
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON) # Make sure C++17 is required
set(CMAKE_CXX_EXTENSIONS OFF)      # Disallow extensions

# Apply the C++ standard to ALL targets. This is a better way to set it.
set_property(TARGET CustomRuntime PROPERTY CXX_STANDARD 17)
set_property(TARGET CustomRuntime PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET CustomRuntime PROPERTY CXX_EXTENSIONS OFF)

if (CMAKE_VERSION VERSION_GREATER 3.12)
    set_property(TARGET CustomRuntime PROPERTY CXX_STANDARD 20)
endif()

# TODO: Add tests and install targets if needed.
