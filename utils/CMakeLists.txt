add_library(ngsapps_utils ${NGS_LIB_TYPE}
  utils_python.cpp randomcf.hpp randomcf.cpp
  # lagrangefespace.hpp lagrangefespace.cpp lagrangefe.hpp lagrangefe.cpp
  myvtkoutput.hpp myvtkoutput.cpp
  composecf.hpp composecf.cpp
  convolutioncf.hpp convolutioncf.cpp
  cachecf.hpp cachecf.cpp
  )

set_target_properties(ngsapps_utils PROPERTIES SUFFIX ".so")

if(NOT WIN32)
    target_link_libraries(ngsapps_utils ${PYTHON_LIBS} ${LAPACK_LIBRARIES} ${NETGEN_LIBS})
  install( TARGETS ngsapps_utils ${ngs_install_dir})
endif(NOT WIN32)

if(USE_PYTHON)
    install (FILES
            utils.py 
            DESTINATION ${PYTHON_PACKAGES_INSTALL_DIR}/ngsapps
            COMPONENT ngsapps
            )
endif(USE_PYTHON)

