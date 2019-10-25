#!/usr/bin/env python

import numpy as np
import cv2
import time
import os
import sys
import glob

from skimage import measure
from plyfile import PlyData, PlyElement

try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    from pycuda.compiler import SourceModule

    FUSION_GPU_MODE = 1
except Exception as err:
    print('Warning: %s' % (str(err)))
    print('Failed to import PyCUDA. Running fusion in CPU mode.')
    FUSION_GPU_MODE = 0


class TSDFVolume(object):

    def __init__(self, vol_bnds, voxel_size):

        # Define voxel volume parameters
        self._vol_bnds = vol_bnds  # rows: x,y,z columns: min,max in world coordinates in meters
        self._voxel_size = voxel_size  # in meters (determines volume discretization and resolution)
        self._trunc_margin = self._voxel_size * 5  # truncation on SDF

        # Adjust volume bounds
        self._vol_dim = np.ceil((self._vol_bnds[:, 1] - self._vol_bnds[:, 0]) / self._voxel_size).copy(
            order='C').astype(int)  # ensure C-order contigous
        self._vol_bnds[:, 1] = self._vol_bnds[:, 0] + self._vol_dim * self._voxel_size
        self._vol_origin = self._vol_bnds[:, 0].copy(order='C').astype(np.float32)  # ensure C-order contigous
        print("Voxel volume size: %d x %d x %d" % (self._vol_dim[0], self._vol_dim[1], self._vol_dim[2]))

        # Initialize pointers to voxel volume in CPU memory
        self._tsdf_vol_cpu = np.ones(self._vol_dim).astype(np.float32)
        self._weight_vol_cpu = np.zeros(self._vol_dim).astype(
            np.float32)  # for computing the cumulative moving average of observations per voxel
        self._color_vol_cpu = np.zeros(self._vol_dim).astype(np.float32)

        # Copy voxel volumes to GPU
        if FUSION_GPU_MODE:
            self._tsdf_vol_gpu = cuda.mem_alloc(self._tsdf_vol_cpu.nbytes)
            cuda.memcpy_htod(self._tsdf_vol_gpu, self._tsdf_vol_cpu)
            self._weight_vol_gpu = cuda.mem_alloc(self._weight_vol_cpu.nbytes)
            cuda.memcpy_htod(self._weight_vol_gpu, self._weight_vol_cpu)
            self._color_vol_gpu = cuda.mem_alloc(self._color_vol_cpu.nbytes)
            cuda.memcpy_htod(self._color_vol_gpu, self._color_vol_cpu)

            # Cuda kernel function (C++)
            self._cuda_src_mod = SourceModule("""
              __global__ void integrate(float * tsdf_vol,
                                        float * weight_vol,
                                        float * color_vol,
                                        float * vol_dim,
                                        float * vol_origin,
                                        float * cam_intr,
                                        float * cam_pose,
                                        float * other_params,
                                        float * color_im,
                                        float * depth_im) {

                // Get voxel index
                int gpu_loop_idx = (int) other_params[0];
                int max_threads_per_block = blockDim.x;
                int block_idx = blockIdx.z*gridDim.y*gridDim.x+blockIdx.y*gridDim.x+blockIdx.x;
                int voxel_idx = gpu_loop_idx*gridDim.x*gridDim.y*gridDim.z*max_threads_per_block+block_idx*max_threads_per_block+threadIdx.x;

                int vol_dim_x = (int) vol_dim[0];
                int vol_dim_y = (int) vol_dim[1];
                int vol_dim_z = (int) vol_dim[2];

                if (voxel_idx > vol_dim_x*vol_dim_y*vol_dim_z)
                    return;

                // Get voxel grid coordinates (note: be careful when casting)
                float voxel_x = floorf(((float)voxel_idx)/((float)(vol_dim_y*vol_dim_z)));
                float voxel_y = floorf(((float)(voxel_idx-((int)voxel_x)*vol_dim_y*vol_dim_z))/((float)vol_dim_z));
                float voxel_z = (float)(voxel_idx-((int)voxel_x)*vol_dim_y*vol_dim_z-((int)voxel_y)*vol_dim_z);

                // Voxel grid coordinates to world coordinates
                float voxel_size = other_params[1];
                float pt_x = vol_origin[0]+voxel_x*voxel_size;
                float pt_y = vol_origin[1]+voxel_y*voxel_size;
                float pt_z = vol_origin[2]+voxel_z*voxel_size;

                // World coordinates to camera coordinates
                float tmp_pt_x = pt_x-cam_pose[0*4+3];
                float tmp_pt_y = pt_y-cam_pose[1*4+3];
                float tmp_pt_z = pt_z-cam_pose[2*4+3];
                float cam_pt_x = cam_pose[0*4+0]*tmp_pt_x+cam_pose[1*4+0]*tmp_pt_y+cam_pose[2*4+0]*tmp_pt_z;
                float cam_pt_y = cam_pose[0*4+1]*tmp_pt_x+cam_pose[1*4+1]*tmp_pt_y+cam_pose[2*4+1]*tmp_pt_z;
                float cam_pt_z = cam_pose[0*4+2]*tmp_pt_x+cam_pose[1*4+2]*tmp_pt_y+cam_pose[2*4+2]*tmp_pt_z;

                // Camera coordinates to image pixels
                int pixel_x = (int) roundf(cam_intr[0*3+0]*(cam_pt_x/cam_pt_z)+cam_intr[0*3+2]);
                int pixel_y = (int) roundf(cam_intr[1*3+1]*(cam_pt_y/cam_pt_z)+cam_intr[1*3+2]);

                // Skip if outside view frustum
                int im_h = (int) other_params[2];
                int im_w = (int) other_params[3];
                if (pixel_x < 0 || pixel_x >= im_w || pixel_y < 0 || pixel_y >= im_h)
                    return;

                // Skip invalid depth
                float depth_value = depth_im[pixel_y*im_w+pixel_x];
                if (depth_value == 0)
                    return;

                // Integrate TSDF
                float trunc_margin = other_params[4];
                float depth_diff = depth_value-cam_pt_z;
                if (depth_diff < -trunc_margin)
                    return;
                float dist = fmin(1.0f,depth_diff/trunc_margin);
                float w_old = weight_vol[voxel_idx];
                float obs_weight = other_params[5];
                float w_new = w_old + obs_weight;
                weight_vol[voxel_idx] = w_new;
                tsdf_vol[voxel_idx] = (tsdf_vol[voxel_idx]*w_old+dist)/w_new;

                // Integrate color
                float old_color = color_vol[voxel_idx];
                float old_b = floorf(old_color/(256*256));
                float old_g = floorf((old_color-old_b*256*256)/256);
                float old_r = old_color-old_b*256*256-old_g*256;
                float new_color = color_im[pixel_y*im_w+pixel_x];
                float new_b = floorf(new_color/(256*256));
                float new_g = floorf((new_color-new_b*256*256)/256);
                float new_r = new_color-new_b*256*256-new_g*256;
                new_b = fmin(roundf((old_b*w_old+new_b)/w_new),255.0f);
                new_g = fmin(roundf((old_g*w_old+new_g)/w_new),255.0f);
                new_r = fmin(roundf((old_r*w_old+new_r)/w_new),255.0f);
                color_vol[voxel_idx] = new_b*256*256+new_g*256+new_r;

              }""")

            self._cuda_integrate = self._cuda_src_mod.get_function("integrate")

            # Determine block/grid size on GPU
            gpu_dev = cuda.Device(0)
            self._max_gpu_threads_per_block = gpu_dev.MAX_THREADS_PER_BLOCK
            n_blocks = int(np.ceil(float(np.prod(self._vol_dim)) / float(self._max_gpu_threads_per_block)))
            grid_dim_x = min(gpu_dev.MAX_GRID_DIM_X, int(np.floor(np.cbrt(n_blocks))))
            grid_dim_y = min(gpu_dev.MAX_GRID_DIM_Y, int(np.floor(np.sqrt(n_blocks / grid_dim_x))))
            grid_dim_z = min(gpu_dev.MAX_GRID_DIM_Z, int(np.ceil(float(n_blocks) / float(grid_dim_x * grid_dim_y))))
            self._max_gpu_grid_dim = np.array([grid_dim_x, grid_dim_y, grid_dim_z]).astype(int)
            self._n_gpu_loops = int(np.ceil(float(np.prod(self._vol_dim)) / float(
                np.prod(self._max_gpu_grid_dim) * self._max_gpu_threads_per_block)))

    # (Deprecated) Expand voxel volume to encompass new bounds
    # def expand(self,new_bnds):
    #     for dim in range(3):
    #         if new_bnds[dim,0] < self._vol_bnds[dim,0]: # expand lower bounds
    #             n_voxels_expand = int(np.ceil((self._vol_bnds[dim,0]-new_bnds[dim,0])/self._voxel_size))
    #             new_chunk_size = np.round((self._vol_bnds[:,1]-self._vol_bnds[:,0])/self._voxel_size).astype(int)
    #             new_chunk_size[dim] = n_voxels_expand # size of expanding region (i.e. chunk)

    #             # Initialize chunks and concatenate to current voxel volume
    #             self._tsdf_vol_cpu = np.concatenate((np.ones(new_chunk_size),self._tsdf_vol_cpu),axis=dim)
    #             self._weight_vol_cpu = np.concatenate((np.zeros(new_chunk_size),self._weight_vol_cpu),axis=dim)
    #             self._color_vol_cpu = np.concatenate((np.zeros(new_chunk_size),self._color_vol_cpu),axis=dim)
    #             self._vol_bnds[dim,0] -= n_voxels_expand*self._voxel_size # update voxel volume bounds

    #         if new_bnds[dim,1] > self._vol_bnds[dim,1]: # expand upper bounds
    #             n_voxels_expand = int(np.ceil((new_bnds[dim,1]-self._vol_bnds[dim,1])/self._voxel_size))
    #             new_chunk_size = np.round((self._vol_bnds[:,1]-self._vol_bnds[:,0])/self._voxel_size).astype(int)
    #             new_chunk_size[dim] = n_voxels_expand # size of expanding region (i.e. chunk)

    #             # Initialize chunks and concatenate to current voxel volume
    #             self._tsdf_vol_cpu = np.concatenate((self._tsdf_vol_cpu,np.ones(new_chunk_size)),axis=dim)
    #             self._weight_vol_cpu = np.concatenate((self._weight_vol_cpu,np.zeros(new_chunk_size)),axis=dim)
    #             self._color_vol_cpu = np.concatenate((self._color_vol_cpu,np.zeros(new_chunk_size)),axis=dim)
    #             self._vol_bnds[dim,1] += n_voxels_expand*self._voxel_size # update voxel volume bounds

    def integrate(self, color_im, depth_im, cam_intr, cam_pose, obs_weight=1.):
        im_h = depth_im.shape[0]
        im_w = depth_im.shape[1]

        # Fold RGB color image into a single channel image
        color_im = color_im.astype(np.float32)
        color_im = np.floor(color_im[:, :, 2] * 256 * 256 + color_im[:, :, 1] * 256 + color_im[:, :, 0])

        # GPU mode: integrate voxel volume (calls CUDA kernel)
        if FUSION_GPU_MODE:
            for gpu_loop_idx in range(self._n_gpu_loops):
                self._cuda_integrate(self._tsdf_vol_gpu,
                                     self._weight_vol_gpu,
                                     self._color_vol_gpu,
                                     cuda.InOut(self._vol_dim.astype(np.float32)),
                                     cuda.InOut(self._vol_origin.astype(np.float32)),
                                     cuda.InOut(cam_intr.reshape(-1).astype(np.float32)),
                                     cuda.InOut(cam_pose.reshape(-1).astype(np.float32)),
                                     cuda.InOut(np.asarray(
                                         [gpu_loop_idx, self._voxel_size, im_h, im_w, self._trunc_margin, obs_weight],
                                         np.float32)),
                                     cuda.InOut(color_im.reshape(-1).astype(np.float32)),
                                     cuda.InOut(depth_im.reshape(-1).astype(np.float32)),
                                     block=(self._max_gpu_threads_per_block, 1, 1), grid=(
                        int(self._max_gpu_grid_dim[0]), int(self._max_gpu_grid_dim[1]), int(self._max_gpu_grid_dim[2])))

        # CPU mode: integrate voxel volume (vectorized implementation)
        else:

            # Get voxel grid coordinates
            xv, yv, zv = np.meshgrid(range(self._vol_dim[0]), range(self._vol_dim[1]), range(self._vol_dim[2]),
                                     indexing='ij')
            vox_coords = np.concatenate((xv.reshape(1, -1), yv.reshape(1, -1), zv.reshape(1, -1)), axis=0).astype(int)

            # Voxel coordinates to world coordinates
            world_pts = self._vol_origin.reshape(-1, 1) + vox_coords.astype(float) * self._voxel_size

            # World coordinates to camera coordinates
            world2cam = np.linalg.inv(cam_pose)
            cam_pts = np.dot(world2cam[:3, :3], world_pts) + np.tile(world2cam[:3, 3].reshape(3, 1),
                                                                     (1, world_pts.shape[1]))

            # Camera coordinates to image pixels
            pix_x = np.round(cam_intr[0, 0] * (cam_pts[0, :] / cam_pts[2, :]) + cam_intr[0, 2]).astype(int)
            pix_y = np.round(cam_intr[1, 1] * (cam_pts[1, :] / cam_pts[2, :]) + cam_intr[1, 2]).astype(int)

            # Skip if outside view frustum
            valid_pix = np.logical_and(pix_x >= 0,
                                       np.logical_and(pix_x < im_w,
                                                      np.logical_and(pix_y >= 0,
                                                                     pix_y < im_h)))

            depth_val = np.zeros(pix_x.shape)
            depth_val[valid_pix] = depth_im[pix_y[valid_pix], pix_x[valid_pix]]

            # Integrate TSDF
            depth_diff = depth_val - cam_pts[2, :]
            valid_pts = np.logical_and(depth_val > 0, depth_diff >= -self._trunc_margin)
            dist = np.minimum(1., np.divide(depth_diff, self._trunc_margin))
            w_old = self._weight_vol_cpu[vox_coords[0, valid_pts], vox_coords[1, valid_pts], vox_coords[2, valid_pts]]
            w_new = w_old + obs_weight
            self._weight_vol_cpu[vox_coords[0, valid_pts], vox_coords[1, valid_pts], vox_coords[2, valid_pts]] = w_new
            tsdf_vals = self._tsdf_vol_cpu[vox_coords[0, valid_pts], vox_coords[1, valid_pts], vox_coords[2, valid_pts]]
            self._tsdf_vol_cpu[
                vox_coords[0, valid_pts], vox_coords[1, valid_pts], vox_coords[2, valid_pts]] = np.divide(
                np.multiply(tsdf_vals, w_old) + dist[valid_pts], w_new)

            # Integrate color
            old_color = self._color_vol_cpu[
                vox_coords[0, valid_pts], vox_coords[1, valid_pts], vox_coords[2, valid_pts]]
            old_b = np.floor(old_color / (256. * 256.))
            old_g = np.floor((old_color - old_b * 256. * 256.) / 256.)
            old_r = old_color - old_b * 256. * 256. - old_g * 256.
            new_color = color_im[pix_y[valid_pts], pix_x[valid_pts]]
            new_b = np.floor(new_color / (256. * 256.))
            new_g = np.floor((new_color - new_b * 256. * 256.) / 256.)
            new_r = new_color - new_b * 256. * 256. - new_g * 256.
            new_b = np.minimum(np.round(np.divide(np.multiply(old_b, w_old) + new_b, w_new)), 255.);
            new_g = np.minimum(np.round(np.divide(np.multiply(old_g, w_old) + new_g, w_new)), 255.);
            new_r = np.minimum(np.round(np.divide(np.multiply(old_r, w_old) + new_r, w_new)), 255.);
            self._color_vol_cpu[vox_coords[0, valid_pts], vox_coords[1, valid_pts], vox_coords[
                2, valid_pts]] = new_b * 256. * 256. + new_g * 256. + new_r;

    # Copy voxel volume to CPU
    def get_volume(self):
        if FUSION_GPU_MODE:
            cuda.memcpy_dtoh(self._tsdf_vol_cpu, self._tsdf_vol_gpu)
            cuda.memcpy_dtoh(self._color_vol_cpu, self._color_vol_gpu)
        return self._tsdf_vol_cpu, self._color_vol_cpu

    # Get mesh of voxel volume via marching cubes
    def get_mesh(self):
        tsdf_vol, color_vol = self.get_volume()

        # Marching cubes
        verts, faces, norms, vals = measure.marching_cubes_lewiner(tsdf_vol, level=0)
        verts_ind = np.round(verts).astype(int)
        verts = verts * self._voxel_size + self._vol_origin  # voxel grid coordinates to world coordinates

        # Get vertex colors
        rgb_vals = color_vol[verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]]
        colors_b = np.floor(rgb_vals / (256 * 256))
        colors_g = np.floor((rgb_vals - colors_b * 256 * 256) / 256)
        colors_r = rgb_vals - colors_b * 256 * 256 - colors_g * 256
        colors = np.floor(np.asarray([colors_r, colors_g, colors_b])).T
        colors = colors.astype(np.uint8)
        return verts, faces, norms, colors


# -------------------------------------------------------------------------------
# Additional helper functions


# Get corners of 3D camera view frustum of depth image
def get_view_frustum(depth_im, cam_intr, cam_pose):
    im_h = depth_im.shape[0]
    im_w = depth_im.shape[1]
    max_depth = np.max(depth_im)
    view_frust_pts = np.array([(np.array([0, 0, 0, im_w, im_w]) - cam_intr[0, 2]) * np.array(
        [0, max_depth, max_depth, max_depth, max_depth]) / cam_intr[0, 0],
                               (np.array([0, 0, im_h, 0, im_h]) - cam_intr[1, 2]) * np.array(
                                   [0, max_depth, max_depth, max_depth, max_depth]) / cam_intr[1, 1],
                               np.array([0, max_depth, max_depth, max_depth, max_depth])])
    view_frust_pts = np.dot(cam_pose[:3, :3], view_frust_pts) + np.tile(cam_pose[:3, 3].reshape(3, 1), (
        1, view_frust_pts.shape[1]))  # from camera to world coordinates
    return view_frust_pts


# Save 3D mesh to a polygon .ply file
def meshwrite_ascii(filename, verts, faces, norms):
    # Write header
    ply_file = open(filename, 'w')
    ply_file.write("ply\n")
    ply_file.write("format ascii 1.0\n")
    ply_file.write("element vertex %d\n" % (verts.shape[0]))
    ply_file.write("property float x\n")
    ply_file.write("property float y\n")
    ply_file.write("property float z\n")
    ply_file.write("property float nx\n")
    ply_file.write("property float ny\n")
    ply_file.write("property float nz\n")
    ply_file.write("element face %d\n" % (faces.shape[0]))
    ply_file.write("property list uchar int vertex_index\n")
    ply_file.write("end_header\n")

    # Write vertex list
    for i in range(verts.shape[0]):
        ply_file.write(
            "%f %f %f %f %f %f\n" % (verts[i, 0], verts[i, 1], verts[i, 2], norms[i, 0], norms[i, 1], norms[i, 2]))

    # Write face list
    for i in range(faces.shape[0]):
        ply_file.write("3 %d %d %d\n" % (faces[i, 0], faces[i, 1], faces[i, 2]))

    ply_file.close()


def meshwrite_color_ascii(filename, verts, faces, norms, colors):
    # Write header
    ply_file = open(filename, 'w')
    ply_file.write("ply\n")
    ply_file.write("format ascii 1.0\n")
    ply_file.write("element vertex %d\n" % (verts.shape[0]))
    ply_file.write("property float x\n")
    ply_file.write("property float y\n")
    ply_file.write("property float z\n")
    ply_file.write("property float nx\n")
    ply_file.write("property float ny\n")
    ply_file.write("property float nz\n")
    ply_file.write("property uchar red\n")
    ply_file.write("property uchar green\n")
    ply_file.write("property uchar blue\n")
    ply_file.write("element face %d\n" % (faces.shape[0]))
    ply_file.write("property list uchar int vertex_index\n")
    ply_file.write("end_header\n")

    # Write vertex list
    for i in range(verts.shape[0]):
        ply_file.write("%f %f %f %f %f %f %d %d %d\n" % (
            verts[i, 0], verts[i, 1], verts[i, 2], norms[i, 0], norms[i, 1], norms[i, 2], colors[i, 0], colors[i, 1],
            colors[i, 2]))

    # Write face list
    for i in range(faces.shape[0]):
        ply_file.write("3 %d %d %d\n" % (faces[i, 0], faces[i, 1], faces[i, 2]))

    ply_file.close()


def meshwrite_binary(filename, verts, faces):
    # print "type(verts): ", type(verts)
    # print "verts.shape: ", verts.shape
    # print "faces.shape: ", faces.shape
    #
    # print "verts[0,:] = ", verts[0, :]
    # print "faces[0,:] = ", faces[0, :]

    # permute faces to get visualization
    # faces = np.flip(faces, 1)

    # converting numpy arrays to format for ply file
    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[('x', 'f4'), ('y', 'f4'),
                                                ('z', 'f4')])
    faces_tuple = np.zeros((num_faces,), dtype=[('vertex_indices', 'i4', (3,))])

    for i in xrange(0, num_verts):
        verts_tuple[i] = tuple(verts[i, :])

    for i in xrange(0, num_faces):
        faces_tuple[i] = faces[i, :].tolist()

    # save it out
    el_verts = PlyElement.describe(verts_tuple, 'vertex')
    el_faces = PlyElement.describe(faces_tuple, 'face')

    ply_data = PlyData([el_verts, el_faces])
    # ply_data = PlyData([el_verts, el_faces], text=True)
    ply_data.write(filename)


def meshwrite_color_binary(filename, verts, faces, norms, colors):
    # print "type(verts): ", type(verts)
    # print "verts.shape: ", verts.shape
    # print "faces.shape: ", faces.shape
    #
    # print "verts[0,:] = ", verts[0, :]
    # print "faces[0,:] = ", faces[0, :]

    # permute faces to get visualization
    # faces = np.flip(faces, 1)

    # converting numpy arrays to format for ply file
    # num_verts = num_colors
    num_verts = verts.shape[0]
    num_colors = colors.shape[0]
    assert num_verts == num_colors
    num_faces = faces.shape[0]
    num_norms = norms.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    faces_tuple = np.zeros((num_faces,), dtype=[('vertex_indices', 'i4', (3,))])
    norms_tuple = np.zeros((num_norms,), dtype=[('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')])
    colors_tuple = np.zeros((num_colors,), dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    for i in xrange(0, num_verts):
        verts_tuple[i] = tuple(verts[i, :])
        colors_tuple[i] = tuple(colors[i, :])

    for i in xrange(0, num_norms):
        norms_tuple[i] = tuple(norms[i, :])

    for i in xrange(0, num_faces):
        faces_tuple[i] = faces[i, :].tolist()

    # merge color to vertex
    vertex_all = np.empty(num_verts, verts_tuple.dtype.descr + norms_tuple.dtype.descr + colors_tuple.dtype.descr)
    for prop in verts_tuple.dtype.names:
        vertex_all[prop] = verts_tuple[prop]

    for prop in norms_tuple.dtype.names:
        vertex_all[prop] = norms_tuple[prop]

    for prop in colors_tuple.dtype.names:
        vertex_all[prop] = colors_tuple[prop]

    # save it out
    el_verts = PlyElement.describe(vertex_all, 'vertex')
    el_faces = PlyElement.describe(faces_tuple, 'face')

    ply_data = PlyData([el_verts, el_faces])
    # ply_data = PlyData([el_verts, el_faces], text=True)
    ply_data.write(filename)


if __name__ == "__main__":
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    print path
    if sys.argv.__len__() > 1:
        path = sys.argv[1]  # get data path

    # get file prefix in data folder
    imagels = sorted(glob.glob(os.path.join(path, '*.png')))
    n_imgs = len(imagels)
    prefixls = []
    for image in imagels:
        prefixls.append(image.split(".")[0])

    # print prefixls

    # prepare for fusion
    max_depth = 1.2  # max depth
    print "Image path to run fusion:", path
    print "Have %d images to fusion..." % n_imgs

    # (Optional) sample code to compute 3D bounds (in world coordinates) around convex hull of all camera view frustums in dataset
    print("Estimating voxel volume bounds...")
    cam_intr = np.loadtxt(path + "/camera-intrinsics.txt", delimiter=' ')
    vol_bnds = np.zeros((3, 2))

    for prefix in prefixls:
        # Read depth image and camera pose
        depth_im = cv2.imread(prefix + ".depth.png", -1).astype(float) / 1000.  # depth is saved in 16-bit PNG in millimeters
        depth_im[depth_im > max_depth] = 0  # set invalid depth to 0 (specific to 7-scenes dataset)
        cam_pose = np.loadtxt(prefix + ".pose.txt")  # 4x4 rigid transformation matrix

        # Compute camera view frustum and extend convex hull
        view_frust_pts = get_view_frustum(depth_im, cam_intr, cam_pose)
        vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))
        vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(view_frust_pts, axis=1))

    # Initialize voxel volume
    print("Initializing voxel volume...")
    tsdf_vol = TSDFVolume(vol_bnds, voxel_size=0.002)

    # Loop through RGB-D images and fuse them together
    t0_elapse = time.time()
    fusion_cnt = 0
    for prefix in prefixls:
        fusion_cnt += 1
        print("Fusing frame %d/%d" % (fusion_cnt, n_imgs))

        # Read RGB-D image and camera pose
        color_image = cv2.cvtColor(cv2.imread(prefix + ".color.jpg"), cv2.COLOR_BGR2RGB)
        depth_im = cv2.imread(prefix + ".depth.png", -1).astype(
            float) / 1000.  # depth is saved in 16-bit PNG in millimeters
        depth_im[depth_im > max_depth] = 0  # set invalid depth to 0 (specific to 7-scenes dataset)
        cam_pose = np.loadtxt(prefix + ".pose.txt")  # 4x4 rigid transformation matrix

        # Integrate observation into voxel volume (assume color aligned with depth)
        tsdf_vol.integrate(color_image, depth_im, cam_intr, cam_pose, obs_weight=1.)

    fps = n_imgs / (time.time() - t0_elapse)
    print("Average FPS: %.2f" % fps)

    # Get mesh from voxel volume and save to disk (can be viewed with Meshlab)
    print("Saving to ply file...")
    verts, faces, norms, colors = tsdf_vol.get_mesh()

    path_save = os.path.dirname(path)
    meshwrite_ascii(path_save + "/mesh_ascii.ply", verts, faces, norms)
    meshwrite_color_ascii(path_save + "/mesh_color_ascii.ply", verts, faces, norms, colors)
    meshwrite_binary(path_save + "/mesh_binary.ply", verts, faces)
    meshwrite_color_binary(path_save + "/mesh_color_binary.ply", verts, faces, norms, colors)
