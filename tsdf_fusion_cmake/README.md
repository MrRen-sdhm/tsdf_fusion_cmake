# tsdf_fusion_cuda_cmake

- tsdf fusion project use cuda、c++11、cmake

- 编译报错：error: token ""__CUDACC_VER__ is no longer supported.
    原因：boost库版本太低
    解决方法1：注释 /usr/local/cuda/include/crt/common_functions.h 第16行：
    #define __CUDACC_VER__ "__CUDACC_VER__ is no longer supported. Use __CUDACC_VER_MAJOR__, __CUDACC_VER_MINOR__, and __CUDACC_VER_BUILD__ instead."
    解决方法2：升级 boost to at least 1.65.1

- 编译报错：/usr/local/include/vtk-7.1/vtkRenderingCoreModule.h:47:0: error: unterminated argument list invoking macro "VTK_AUTOINIT1"
    解决办法：在CmakList.txt中添加：
    get_directory_property(dir_defs DIRECTORY ${CMAKE_SOURCE_DIR} COMPILE_DEFINITIONS)
    #set(vtk_flags)
    #foreach(it ${dir_defs})
    #    if(it MATCHES "vtk*")
    #        list(APPEND vtk_flags ${it})
    #    endif()
    #endforeach()
    #
    #foreach(d ${vtk_flags})
    #    remove_definitions(-D${d})
    #endforeach()