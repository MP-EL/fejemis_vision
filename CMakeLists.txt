cmake_minimum_required(VERSION 3.0.2)
project(fejemis_vision)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  message_filters
  cv_bridge
  image_transport
)

find_package (OpenCV 4.0.0 REQUIRED)
include_directories ("/usr/include/opencv4/")

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES fejemis
#  CATKIN_DEPENDS roscpp rospy std_msgs
#  DEPENDS system_lib
)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

## Declare a C++ library
# add_library(${PROJECT_NAME}
#   src/${PROJECT_NAME}/fejemis.cpp
# )

set(fejemis_SRCS
      #
)


#add_executable(main_Rosnode ${fejemis_SRCS})
#add_dependencies(main_Rosnode ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
#target_link_libraries(main_Rosnode ${catkin_LIBRARIES}
#                                   ${OpenCV_LIBRARIES})

