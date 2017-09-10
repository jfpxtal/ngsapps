add_ngsolve_python_module(libngsapps_utils
  utils_python.cpp randomcf.hpp randomcf.cpp
  lagrangefespace.hpp lagrangefespace.cpp lagrangefe.hpp lagrangefe.cpp
  myvtkoutput.hpp myvtkoutput.cpp
  composecf.hpp composecf.cpp
  parameterlf.hpp parameterlf.cpp
  convolutioncf.hpp convolutioncf.cpp
  cachecf.hpp cachecf.cpp
  zlogzcf.hpp zlogzcf.cpp
  annulusspeedcf.hpp annulusspeedcf.cpp
  utils.hpp utils.cpp
  limiter.hpp limiter.cpp
  )

install(TARGETS libngsapps_utils DESTINATION .)

install (FILES
        utils.py meshtools.py merge_meshes.py plotting.py limiter.py
        DESTINATION .
        )
