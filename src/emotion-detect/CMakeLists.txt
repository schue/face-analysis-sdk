# -*-cmake-*-
add_executable(emotion-detect
  main.cpp)
target_link_libraries(emotion-detect
  utilities
  clmTracker
  ${LIBS})
