cmake_minimum_required(VERSION 2.8)

set(COLOR_LIB_SRC colotracker_.cpp histogram_.cpp region_.cpp)

add_library(color_ STATIC ${COLOR_LIB_SRC})
target_link_libraries(color_ ${OpenCV_LIBS})
set_target_properties(color_ PROPERTIES VERSION 1.0.0 SOVERSION 1)
