from utilities2015 import *
from metadata import *


from itertools import groupby

def contours_to_volume(contours_grouped_by_label=None, label_contours_tuples=None, interpolation_direction='z',
                      return_shell=False, len_interval=20):
    """
    Return volume as 3D array, and origin (xmin,xmax,ymin,ymax,zmin,zmax)
    """
    
    import sys
    sys.path.append(os.environ['REPO_DIR'] + '/utilities')
    from annotation_utilities import interpolate_contours_to_volume
    
    
    if label_contours_tuples is not None:
        contours_grouped_by_label = {}
        for label, contours in groupby(contour_label_tuples, key=lambda l, cnts: l):
            contours_grouped_by_label[label] = contours
    else:
        assert contours_grouped_by_label is not None
            
    if isinstance(contours_grouped_by_label.values()[0], dict):
        # dict value is contours grouped by z
        if interpolation_direction == 'z':
            contours_xyz_grouped_by_label = {label: [(x,y,z) for z, (x,y) in contours_grouped.iteritems()]
                            for label, contours_grouped in contours_grouped_by_label.iteritems()}
        elif interpolation_direction == 'y':
            contours_xyz_grouped_by_label = {label: [(x,y,z) for y, (x,z) in contours_grouped.iteritems()]
                            for label, contours_grouped in contours_grouped_by_label.iteritems()}
        elif interpolation_direction == 'x':
            contours_xyz_grouped_by_label = {label: [(x,y,z) for x, (y,z) in contours_grouped.iteritems()]
                            for label, contours_grouped in contours_grouped_by_label.iteritems()}
        
    else:
        contours_xyz_grouped_by_label = contours_grouped_by_label
        # dict value is list of (x,y,z) tuples
#         contours_grouped_by_label = {groupby(contours_xyz, lambda x,y,z: z) 
#                                      for label, contours_xyz in contours_grouped_by_label.iteritems()}
#         pass
            
    xyz_max = [0, 0, 0]
    xyz_min = [np.inf, np.inf, np.inf]
    for label, contours in contours_xyz_grouped_by_label.iteritems():
        xyz_max = np.maximum(xyz_max, np.max(np.vstack(contours), axis=0))
        xyz_min = np.minimum(xyz_min, np.min(np.vstack(contours), axis=0))
        
    xmin, ymin, zmin = np.floor(xyz_min).astype(np.int)
    xmax, ymax, zmax = np.ceil(xyz_max).astype(np.int)
    xdim, ydim, zdim = xmax+1-xmin, ymax+1-ymin, zmax+1-zmin
    
    
    volume = np.zeros((ydim, xdim, zdim), np.uint8)
    
    if return_shell:
        
        for label, contours in contours_grouped_by_label.iteritems():
            
            voxels_grouped = interpolate_contours_to_volume(interpolation_direction=interpolation_direction, 
                                                            contours_xyz=contours, return_contours=True,
                                                            len_interval=len_interval)

            if interpolation_direction == 'z':
                for z, xys in voxels_grouped.iteritems():
                    volume[xys[:,1]-ymin, xys[:,0]-xmin, z-zmin] = label
            elif interpolation_direction == 'y':
                for y, xzs in voxels_grouped.iteritems():
                    volume[y-ymin, xzs[:,0]-xmin, xzs[:,1]-zmin] = label
            elif interpolation_direction == 'x':
                for x, yzs in voxels_grouped.iteritems():
                    volume[yzs[:,0]-ymin, x-xmin, yzs[:,1]-zmin] = label

        return volume, (xmin,xmax,ymin,ymax,zmin,zmax)

    else:
    
        for label, contours in contours_grouped_by_label.iteritems():
            
            voxels_grouped = interpolate_contours_to_volume(interpolation_direction=interpolation_direction, 
                                                                 contours_xyz=contours, return_voxels=True)

            if interpolation_direction == 'z':
                for z, xys in voxels_grouped.iteritems():
                    volume[xys[:,1]-ymin, xys[:,0]-xmin, z-zmin] = label
            elif interpolation_direction == 'y':
                for y, xzs in voxels_grouped.iteritems():
                    volume[y-ymin, xzs[:,0]-xmin, xzs[:,1]-zmin] = label
            elif interpolation_direction == 'x':
                for x, yzs in voxels_grouped.iteritems():
                    volume[yzs[:,0]-ymin, x-xmin, yzs[:,1]-zmin] = label

        return volume, (xmin,xmax,ymin,ymax,zmin,zmax)
    
    

def volume_to_images(volume, voxel_size, cut_dimension, pixel_size=None):

    volume_shape = volume.shape

    if pixel_size is None:
        pixel_size = min(voxel_size)

    if cut_dimension == 0:
        volume_shape01 = volume_shape[1], volume_shape[2]
        voxel_size01 = voxel_size[1], voxel_size[2]
    elif cut_dimension == 1:
        volume_shape01 = volume_shape[0], volume_shape[2]
        voxel_size01 = voxel_size[0], voxel_size[2]
    elif cut_dimension == 2:
        volume_shape01 = volume_shape[0], volume_shape[1]
        voxel_size01 = voxel_size[0], voxel_size[1]

    volume_dim01 = volume_shape01[0] * voxel_size01[0], volume_shape01[1] * voxel_size01[1]
    sample_voxels_0 = np.arange(0, volume_dim01[0], pixel_size) / voxel_size01[0]
    sample_voxels_1 = np.arange(0, volume_dim01[1], pixel_size) / voxel_size01[1]

    if cut_dimension == 0:
        images = volume[:, sample_voxels_0[:,None], sample_voxels_1]
    elif cut_dimension == 1:
        images = volume[sample_voxels_0[:,None], :, sample_voxels_1]
    elif cut_dimension == 2:
        images = volume[sample_voxels_0[:,None], sample_voxels_1, :]

    return images

