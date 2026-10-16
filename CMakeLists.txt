cmake_minimum_required(VERSION 3.20)
project(vs-frameblender VERSION 3.2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release)
endif()

# x86 only. on aarch64 (apple silicon, arm linux) neon is the baseline, so the three variants
# build identically and the dispatch always picks baseline - correct, just slightly redundant.
# CMAKE_OSX_ARCHITECTURES wins on mac, where CMAKE_SYSTEM_PROCESSOR reports the host.
set(FB_ISA_VARIANTS OFF)
if(CMAKE_OSX_ARCHITECTURES)
    if(CMAKE_OSX_ARCHITECTURES MATCHES "x86_64" AND NOT CMAKE_OSX_ARCHITECTURES MATCHES "arm64")
        set(FB_ISA_VARIANTS ON)
    endif()
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|i[3-6]86|x86)$")
    set(FB_ISA_VARIANTS ON)
endif()
message(STATUS "frameblender: per-ISA kernel variants: ${FB_ISA_VARIANTS}")

# the kernel is compiled once per instruction set and picked by cpuid at load, the same shape
# upstream vapoursynth uses for its own filters. -ffast-math is deliberately never added: the
# rounding in kernel.h depends on the add/subtract pair surviving.
function(fb_add_kernel target source level)
    add_library(${target} OBJECT ${source})
    target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    set_property(TARGET ${target} PROPERTY POSITION_INDEPENDENT_CODE ON)

    if(MSVC)
        target_compile_options(${target} PRIVATE /O2)
        if(FB_ISA_VARIANTS)
            if(level STREQUAL "avx2")
                target_compile_options(${target} PRIVATE /arch:AVX2)
            elseif(level STREQUAL "avx512")
                target_compile_options(${target} PRIVATE /arch:AVX512)
            endif()
        endif()
    else()
        target_compile_options(${target} PRIVATE -O3)
        if(FB_ISA_VARIANTS)
            if(level STREQUAL "avx2")
                target_compile_options(${target} PRIVATE -mavx2 -mfma)
            elseif(level STREQUAL "avx512")
                target_compile_options(${target} PRIVATE -mavx512f -mavx512bw -mavx512vl -mavx512dq)
            endif()
        endif()
    endif()
endfunction()

fb_add_kernel(fb_kernel_baseline src/kernel_baseline.cpp baseline)
fb_add_kernel(fb_kernel_avx2     src/kernel_avx2.cpp     avx2)
fb_add_kernel(fb_kernel_avx512   src/kernel_avx512.cpp   avx512)

add_library(frameblender SHARED
    src/plugin.cpp
    $<TARGET_OBJECTS:fb_kernel_baseline>
    $<TARGET_OBJECTS:fb_kernel_avx2>
    $<TARGET_OBJECTS:fb_kernel_avx512>
)

target_include_directories(frameblender PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_definitions(frameblender PRIVATE
    FB_VERSION_MAJOR=${PROJECT_VERSION_MAJOR} FB_VERSION_MINOR=${PROJECT_VERSION_MINOR})
set_target_properties(frameblender PROPERTIES PREFIX "" OUTPUT_NAME "frameblender")

if(MSVC)
    target_compile_options(frameblender PRIVATE /O2)
else()
    target_compile_options(frameblender PRIVATE -O3)
endif()

if(SKBUILD)
    # the wheel install dir is already vapoursynth/plugins, which vapoursynth autoloads from
    install(TARGETS frameblender
        LIBRARY DESTINATION .
        RUNTIME DESTINATION .
        ARCHIVE DESTINATION . EXCLUDE_FROM_ALL)
else()
    install(TARGETS frameblender LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
endif()
