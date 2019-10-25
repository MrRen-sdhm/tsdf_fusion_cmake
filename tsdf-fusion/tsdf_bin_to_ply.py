import os
import subprocess
import shutil
import numpy as np
import math
import yaml
import time
from skimage import measure
from plyfile import PlyData, PlyElement
import array

def convert_tsdf_to_ply(tsdf_bin_filename, tsdf_mesh_filename):
    """
    Converts the tsdf binary file to a mesh file in ply format

    The indexing in the tsdf is
    (x,y,z) <--> (x + y * dim_x + z * dim_x * dim_y)
    """
    start_time = time.time()
    fin = open(tsdf_bin_filename, "rb")

    tsdfHeader = array.array("f")  # f is the typecode for float32
    tsdfHeader.fromfile(fin, 8)
    # print tsdfHeader
    # print type(tsdfHeader)

    voxelGridDim = tsdfHeader[0:3]
    voxelGridDim = np.asarray(voxelGridDim, dtype=np.int)
    voxelGridOrigin = tsdfHeader[3:6]
    voxelSize = tsdfHeader[6]
    truncMargin = tsdfHeader[7]

    dim_x = voxelGridDim[0]
    dim_y = voxelGridDim[1]
    dim_z = voxelGridDim[2]


    headerSize = 8
    tsdf_vec = np.fromfile(tsdf_bin_filename, np.float32)
    tsdf_vec = tsdf_vec[headerSize:]
    tsdf = np.reshape(tsdf_vec, voxelGridDim, order='F') # reshape using Fortran order

    # for loop version of the above reshape operation
    # for x in xrange(0, dim_x):
    #     for y in xrange(0, dim_y):
    #         for z in xrange(0, dim_z):
    #             idx = x + y * dim_x + z * dim_x * dim_y
    #             tsdf[x,y,z] = tsdf_vec[idx]


    print "tsdf.shape:", tsdf.shape
    print "voxelGridDim: ", voxelGridDim
    print "voxeGridOrigin: ", voxelGridOrigin
    print "tsdf.shape:", tsdf.shape

    verts, faces, normals, values = measure.marching_cubes_lewiner(tsdf, spacing=[voxelSize]*3, level=0)


    print "type(verts): ", type(verts)
    print "verts.shape: ", verts.shape
    print "faces.shape:", faces.shape

    print "np.max(verts[:,0]): ", np.max(verts[:,0])
    print "np.min(verts[:,0]): ", np.min(verts[:, 0])


    print "verts[0,:] = ", verts[0,:]
    print "faces[0,:] = ", faces[0,:]

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    # mesh_points = verts
    mesh_points[:,0] = voxelGridOrigin[0] + verts[:,0]
    mesh_points[:,1] = voxelGridOrigin[1] + verts[:,1]
    mesh_points[:,2] = voxelGridOrigin[2] + verts[:,2]

    # permute faces to get visualization
    # faces = np.flip(faces, 1)


    # try writing to the ply file
    print "converting numpy arrays to format for ply file"
    ply_conversion_start_time = time.time()

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[('x', 'f4'), ('y', 'f4'),
                                                ('z', 'f4')])
    faces_tuple = np.zeros((num_faces,), dtype=[('vertex_indices', 'i4', (3,))])

    for i in xrange(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    for i in xrange(0, num_faces):
        faces_tuple[i] = faces[i, :].tolist()



    # save it out
    # try to save it
    el_verts = PlyElement.describe(verts_tuple, 'vertex')
    el_faces = PlyElement.describe(faces_tuple, 'face')

    ply_data = PlyData([el_verts, el_faces])
    print "saving mesh to %s" %(tsdf_mesh_filename)
    ply = ply_data.write(tsdf_mesh_filename)

    print "converting to ply format and writing to file took", time.time() - start_time


convert_tsdf_to_ply("./tsdf.bin", "./tsdf.ply")
