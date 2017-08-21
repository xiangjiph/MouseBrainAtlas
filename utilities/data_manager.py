import sys
import os
import subprocess

from pandas import read_hdf

sys.path.append(os.path.join(os.environ['REPO_DIR'], 'utilities'))
from utilities2015 import *
from metadata import *
from vis3d_utilities import *
from distributed_utilities import *

def get_random_masked_regions(region_shape, stack, num_regions=1, sec=None, fn=None):
    """
    Return a random region that is on mask.

    Args:
        region_shape ((width, height)-tuple):

    Returns:
        list of (region_x, region_y, region_w, region_h)
    """

    if fn is None:
        fn = metadata_cache['sections_to_filenames'][stack][sec]
    tb_mask = DataManager.load_thumbnail_mask_v2(stack=stack, fn=fn)
    img_w, img_h = metadata_cache['image_shape'][stack]
    h, w = region_shape

    regions = []
    for _ in range(num_regions):
        while True:
            xmin = np.random.randint(0, img_w, 1)[0]
            ymin = np.random.randint(0, img_h, 1)[0]

            if xmin + w >= img_w or ymin + h >= img_h:
                continue

            tb_xmin = xmin / 32
            tb_xmax = (xmin + w) / 32
            tb_ymin = ymin / 32
            tb_ymax = (ymin + h) / 32

            if np.count_nonzero(np.r_[tb_mask[tb_ymin, tb_xmin], \
                                      tb_mask[tb_ymin, tb_xmax], \
                                      tb_mask[tb_ymax, tb_xmin], \
                                      tb_mask[tb_ymax, tb_xmax]]) >= 3:
                break
        regions.append((xmin, ymin, w, h))

    return regions

def is_invalid(fn=None, sec=None, stack=None):
    if sec is not None:
        assert stack is not None
        fn = metadata_cache['sections_to_filenames'][stack][sec]
    else:
        assert fn is not None
    return fn in ['Nonexisting', 'Rescan', 'Placeholder']

def volume_type_to_str(t):
    if t == 'score':
        return 'scoreVolume'
    elif t == 'annotation':
        return 'annotationVolume'
    elif t == 'annotationAsScore':
        return 'annotationAsScoreVolume'
    elif t == 'outer_contour':
        return 'outerContourVolume'
    elif t == 'intensity':
        return 'intensityVolume'
    elif t == 'intensity_metaimage':
        return 'intensityMetaImageVolume'
    else:
        raise Exception('Volume type %s is not recognized.' % t)

# def generate_suffix(train_sample_scheme=None, global_transform_scheme=None, local_transform_scheme=None):

#     suffix = []
#     if train_sample_scheme is not None:
#         suffix.append('trainSampleScheme_%d'%train_sample_scheme)
#     if global_transform_scheme is not None:
#         suffix.append('globalTxScheme_%d'%global_transform_scheme)
#     if local_transform_scheme is not None:
#         suffix.append('localTxScheme_%d'%local_transform_scheme)

#     return '_'.join(suffix)


class DataManager(object):

    ########################
    ##   Lauren's data    ##
    ########################
    
    @staticmethod
    def get_lauren_markers_filepath(stack, structure):
        return os.path.join('/shared/lauren_data/', 'markers', stack, stack + '_markers_%s.bp' % structure)
    
    ##############
    ##   SPM    ##
    ##############
    
    @staticmethod
    def get_spm_histograms_filepath(stack, level, section=None, fn=None):
        """
        Args:
            level (int): 0, 1 or 2
        """
        if fn is None:
            fn = metadata_cache['sections_to_filenames'][stack][section]
        return os.path.join(CSHL_SPM_ROOTDIR, 'sift_histograms', stack, fn + '_sift_histograms_l%d.bp' % level)
    
    @staticmethod
    def get_sift_descriptor_vocabulary_filepath():
        """
        Return a sklearn.KMeans classifier object.
        """
        return os.path.join(CSHL_SPM_ROOTDIR, 'sift_vocabulary.clf')
   
    @staticmethod
    def get_sift_descriptors_labelmap_filepath(stack, section=None, fn=None):
        if fn is None:
            fn = metadata_cache['sections_to_filenames'][stack][section]
        return os.path.join(CSHL_SPM_ROOTDIR, 'sift_labelmap', stack, fn + '_sift_labelmap.bp')
   
    @staticmethod
    def get_sift_descriptors_filepath(stack, section=None, fn=None):
        if fn is None:
            fn = metadata_cache['sections_to_filenames'][stack][section]
        return os.path.join(CSHL_SPM_ROOTDIR, 'sift_descriptors', stack, fn + '_sift_descriptors.bp')
   
    @staticmethod
    def get_sift_keypoints_filepath(stack, section=None, fn=None):
        if fn is None:
            fn = metadata_cache['sections_to_filenames'][stack][section]
        return os.path.join(CSHL_SPM_ROOTDIR, 'sift_keypoints', stack, fn + '_sift_keypoints.bp')
    

    ##########################
    ###    Annotation    #####
    ##########################

    @staticmethod
    def get_structure_pose_corrections(stack, stack_m=None,
                                classifier_setting_m=None,
                                classifier_setting_f=None,
                                warp_setting=None, trial_idx=None):
        basename = DataManager.get_warped_volume_basename(**locals())
        fp = os.path.join(ANNOTATION_ROOTDIR, stack, basename + '_' + 'structure3d_corrections' + '.pkl')
        return fp

    @staticmethod
    def get_annotated_structures(stack):
        """
        Return existing structures on every section in annotation.
        """
        contours, _ = load_annotation_v3(stack, annotation_rootdir=ANNOTATION_ROOTDIR)
        annotated_structures = {sec: list(set(contours[contours['section']==sec]['name']))
                                for sec in range(first_sec, last_sec+1)}
        return annotated_structures

    @staticmethod
    def load_annotation_to_grid_indices_lookup(stack, win_id, by_human, stack_m=None,
                                classifier_setting_m=None,
                                classifier_setting_f=None,
                                warp_setting=None, trial_idx=None):

        grid_indices_lookup_fp = DataManager.get_annotation_to_grid_indices_lookup_filepath(**locals())
        download_from_s3(grid_indices_lookup_fp)

        if not os.path.exists(grid_indices_lookup_fp):
            raise Exception("Do not find structure to grid indices lookup file. Please generate it using `generate_annotation_to_grid_indices_lookup`")
        else:
            grid_indices_lookup = load_hdf_v2(grid_indices_lookup_fp)
            # grid_indices_lookup = read_hdf(grid_indices_lookup_fp, 'grid_indices')
            return grid_indices_lookup

    @staticmethod
    def get_annotation_to_grid_indices_lookup_filepath(stack, win_id, by_human, stack_m=None,
                                classifier_setting_m=None,
                                classifier_setting_f=None,
                                warp_setting=None, trial_idx=None):
        if by_human:
            fp = os.path.join(ANNOTATION_ROOTDIR, stack, '%(stack)s_annotation_win%(win)d_grid_indices_lookup.hdf' % {'stack':stack, 'win':win_id})
        else:
            basename = DataManager.get_warped_volume_basename(stack_m=stack_m, stack_f=stack,
                                                              classifier_setting_m=classifier_setting_m,
                                                              classifier_setting_f=classifier_setting_f,
                                                              warp_setting=warp_setting, trial_idx=trial_idx)
            fp = os.path.join(ANNOTATION_ROOTDIR, stack, 'annotation_%(basename)s_win%(win)d_grid_indices_lookup.hdf' % {'basename': basename, 'win':win_id})
        return fp

    @staticmethod
    def get_annotation_filepath(stack, by_human, stack_m=None,
                                classifier_setting_m=None,
                                classifier_setting_f=None,
                                warp_setting=None, trial_idx=None, suffix=None, timestamp=None):
        """
        Args:
            timestamp (str): can be "latest".
        """
        
        
        if by_human:
            # if suffix is None:                
            #     fp = os.path.join(ANNOTATION_ROOTDIR, stack, '%(stack)s_annotation_v3.h5' % {'stack':stack})
            # else:
            if timestamp is not None:
                if timestamp == 'latest':
                    download_from_s3(os.path.join(ANNOTATION_ROOTDIR, stack), is_dir=True, include_only="*contours*", redownload=True)
                    import re
                    timestamps = []
                    for fn in os.listdir(os.path.join(ANNOTATION_ROOTDIR, stack)):
                        m = re.match('%(stack)s_annotation_%(suffix)s_(.*?).hdf' % {'stack':stack, 'suffix': suffix}, fn)
                        if m is not None:
                            ts = m.groups()[0]
                            try:
                                timestamps.append((datetime.strptime(ts, '%m%d%Y%H%M%S'), ts))
                            except:
                                pass
                    timestamp = sorted(timestamps)[-1][1]
                    print "latest timestamp: ", timestamp
                    
                fp = os.path.join(ANNOTATION_ROOTDIR, stack, '%(stack)s_annotation_%(suffix)s_%(timestamp)s.hdf' % {'stack':stack, 'suffix':suffix, 'timestamp': timestamp})
            else:
                fp = os.path.join(ANNOTATION_ROOTDIR, stack, '%(stack)s_annotation_%(suffix)s.hdf' % {'stack':stack, 'suffix':suffix})
        else:
            basename = DataManager.get_warped_volume_basename(stack_m=stack_m, stack_f=stack,
                                                              classifier_setting_m=classifier_setting_m,
                                                              classifier_setting_f=classifier_setting_f,
                                                              warp_setting=warp_setting, trial_idx=trial_idx)
            if suffix is not None:
                if timestamp is not None:
                    fp = os.path.join(ANNOTATION_ROOTDIR, stack, 'annotation_%(basename)s_%(suffix)s_%(timestamp)s.hdf' % {'basename': basename, 'suffix': suffix, 'timestamp': timestamp})
                else:
                    fp = os.path.join(ANNOTATION_ROOTDIR, stack, 'annotation_%(basename)s_%(suffix)s.hdf' % {'basename': basename, 'suffix': suffix})
            else:
                fp = os.path.join(ANNOTATION_ROOTDIR, stack, 'annotation_%(basename)s.hdf' % {'basename': basename})
        return fp

    @staticmethod
    def load_annotation_v4(stack=None, by_human=True, stack_m=None,
                                classifier_setting_m=None,
                                classifier_setting_f=None,
                                warp_setting=None, trial_idx=None, timestamp=None, suffix=None):
        if by_human:
            fp = DataManager.get_annotation_filepath(stack, by_human=True, suffix=suffix, timestamp=timestamp)
            download_from_s3(fp)
            contour_df = read_hdf(fp)
            return contour_df

        else:
            fp = DataManager.get_annotation_filepath(stack, by_human=False,
                                                     stack_m=stack_m,
                                                      classifier_setting_m=classifier_setting_m,
                                                      classifier_setting_f=classifier_setting_f,
                                                      warp_setting=warp_setting, trial_idx=trial_idx,
                                                    suffix=suffix, timestamp=timestamp)
            download_from_s3(fp)
            annotation_df = load_hdf_v2(fp)
            return annotation_df


    # @staticmethod
    # def load_annotation_v3(stack=None, by_human=True, stack_m=None,
    #                             classifier_setting_m=None,
    #                             classifier_setting_f=None,
    #                             warp_setting=None, trial_idx=None, timestamp=None, suffix=None):
    #     if by_human:
    #         # try:
    #         #     structure_df = read_hdf(fp, 'structures')
    #         # except Exception as e:
    #         #     print e
    #         #     sys.stderr.write('Annotation has no structures.\n')
    #         #     return contour_df, None
    #         #
    #         # sys.stderr.write('Loaded annotation %s.\n' % fp)
    #         # return contour_df, structure_df
    #         raise Exception('Not implemented.')
    #     else:
    #         fp = DataManager.get_annotation_filepath(stack, by_human=False,
    #                                                  stack_m=stack_m,
    #                                                   classifier_setting_m=classifier_setting_m,
    #                                                   classifier_setting_f=classifier_setting_f,
    #                                                   warp_setting=warp_setting, trial_idx=trial_idx,
    #                                                 suffix=suffix, timestamp=timestamp)
    #         download_from_s3(fp)
    #         annotation_df = load_hdf_v2(fp)
    #         return annotation_df

    @staticmethod
    def get_annotation_viz_dir(stack):
        return os.path.join(ANNOTATION_VIZ_ROOTDIR, stack)

    @staticmethod
    def get_annotation_viz_filepath(stack, sec=None, fn=None):
        if fn is None:
            fn = metadata_cache['sections_to_filenames'][sec]
        return os.path.join(ANNOTATION_VIZ_ROOTDIR, stack, fn + '_annotation_viz.tif')


    ########################################################

    @staticmethod
    def load_data(filepath, filetype):

        if not os.path.exists(filepath):
            sys.stderr.write('File does not exist: %s\n' % filepath)

        if filetype == 'bp':
            return bp.unpack_ndarray_file(filepath)
        elif filetype == 'image':
            return imread(filepath)
        elif filetype == 'hdf':
            try:
                return load_hdf(filepath)
            except:
                return load_hdf_v2(filepath)
        elif filetype == 'bbox':
            return np.loadtxt(filepath).astype(np.int)
        elif filetype == 'annotation_hdf':
            contour_df = read_hdf(filepath, 'contours')
            return contour_df
        elif filetype == 'pickle':
            import cPickle as pickle
            return pickle.load(open(filepath, 'r'))
        elif filetype == 'file_section_map':
            with open(filepath, 'r') as f:
                fn_idx_tuples = [line.strip().split() for line in f.readlines()]
                filename_to_section = {fn: int(idx) for fn, idx in fn_idx_tuples}
                section_to_filename = {int(idx): fn for fn, idx in fn_idx_tuples}
            return filename_to_section, section_to_filename
        elif filetype == 'label_name_map':
            label_to_name = {}
            name_to_label = {}
            with open(filepath, 'r') as f:
                for line in f.readlines():
                    name_s, label = line.split()
                    label_to_name[int(label)] = name_s
                    name_to_label[name_s] = int(label)
            return label_to_name, name_to_label
        elif filetype == 'anchor':
            with open(filepath, 'r') as f:
                anchor_fn = f.readline()
            return anchor_fn
        elif filetype == 'transform_params':
            with open(filepath, 'r') as f:
                lines = f.readlines()

                global_params = one_liner_to_arr(lines[0], float)
                centroid_m = one_liner_to_arr(lines[1], float)
                xdim_m, ydim_m, zdim_m  = one_liner_to_arr(lines[2], int)
                centroid_f = one_liner_to_arr(lines[3], float)
                xdim_f, ydim_f, zdim_f  = one_liner_to_arr(lines[4], int)

            return global_params, centroid_m, centroid_f, xdim_m, ydim_m, zdim_m, xdim_f, ydim_f, zdim_f
        else:
            sys.stderr.write('File type %s not recognized.\n' % filetype)

    @staticmethod
    def get_anchor_filename_filename(stack):
        fn = os.path.join(THUMBNAIL_DATA_DIR, stack, stack + '_anchor.txt')
        return fn

    @staticmethod
    def load_anchor_filename(stack):
        fp = DataManager.get_anchor_filename_filename(stack)
        download_from_s3(fp, local_root=THUMBNAIL_DATA_ROOTDIR)
        anchor_fn = DataManager.load_data(fp, filetype='anchor')
        return anchor_fn

    @staticmethod
    def get_cropbox_filename(stack, anchor_fn=None):
        if anchor_fn is None:
            anchor_fn = DataManager.load_anchor_filename(stack=stack)
        fn = os.path.join(THUMBNAIL_DATA_DIR, stack, stack + '_alignedTo_' + anchor_fn + '_cropbox.txt')
        return fn

    @staticmethod
    def load_cropbox(stack, anchor_fn=None, convert_section_to_z=False):
        """
        Args:
            convert_section_to_z (bool): If true, return (xmin,xmax,ymin,ymax,zmin,zmax); if false, return (xmin,xmax,ymin,ymax,secmin,secmax)
        """
        
        fp = DataManager.get_cropbox_filename(stack=stack, anchor_fn=anchor_fn)
        download_from_s3(fp, local_root=THUMBNAIL_DATA_ROOTDIR)
        
        if convert_section_to_z:
            xmin, xmax, ymin, ymax, secmin, secmax = np.loadtxt(fp).astype(np.int)
            zmin = int(np.mean(DataManager.convert_section_to_z(stack=stack, sec=secmin, downsample=32, z_begin=0)))
            zmax = int(np.mean(DataManager.convert_section_to_z(stack=stack, sec=secmax, downsample=32, z_begin=0)))
            cropbox = np.array((xmin, xmax, ymin, ymax, zmin, zmax))
        else:    
            cropbox = np.loadtxt(fp).astype(np.int)
        return cropbox

    @staticmethod
    def get_sorted_filenames_filename(stack):
        fn = os.path.join(THUMBNAIL_DATA_DIR, stack, stack + '_sorted_filenames.txt')
        return fn

    @staticmethod
    def load_sorted_filenames(stack):
        """
        Get the mapping between section index and image filename.

        Returns:
            Two dicts: filename_to_section, section_to_filename
        """

        fp = DataManager.get_sorted_filenames_filename(stack)
        download_from_s3(fp, local_root=THUMBNAIL_DATA_ROOTDIR)
        filename_to_section, section_to_filename = DataManager.load_data(fp, filetype='file_section_map')
        if 'Placeholder' in filename_to_section:
            filename_to_section.pop('Placeholder')
        return filename_to_section, section_to_filename

    @staticmethod
    def get_transforms_filename(stack, anchor_fn=None):
        if anchor_fn is None:
            anchor_fn = metadata_cache['anchor_fn'][stack]
        fn = os.path.join(THUMBNAIL_DATA_DIR, stack, stack + '_transformsTo_%s.pkl' % anchor_fn)
        return fn

    @staticmethod
    def load_transforms(stack, downsample_factor, use_inverse=True, anchor_fn=None):
        """
        Args:
            use_inverse (bool): If True, load the transforms that when multiplied
            to a point on original space converts it to on aligned space.
            In preprocessing, set to False, which means simply parse the transform files as they are.
        """

        fp = DataManager.get_transforms_filename(stack, anchor_fn=anchor_fn)
        download_from_s3(fp, local_root=THUMBNAIL_DATA_ROOTDIR)
        Ts = DataManager.load_data(fp, filetype='pickle')

        if use_inverse:
            Ts_inv_downsampled = {}
            for fn, T0 in Ts.iteritems():
                T = T0.copy()
                T[:2, 2] = T[:2, 2] * 32 / downsample_factor
                Tinv = np.linalg.inv(T)
                Ts_inv_downsampled[fn] = Tinv
            return Ts_inv_downsampled
        else:
            Ts_downsampled = {}
            for fn, T0 in Ts.iteritems():
                T = T0.copy()
                T[:2, 2] = T[:2, 2] * 32 / downsample_factor
                Ts_downsampled[fn] = T
            return Ts_downsampled

    ################
    # Registration #
    ################

    # @staticmethod
    # def get_original_volume_basename(stack, classifier_setting=None, downscale=32, volume_type='score', **kwargs):
    #     return DataManager.get_warped_volume_basename(stack_m=stack, classifier_setting_m=classifier_setting,
    #     downscale=downscale, type_m=volume_type)

    @staticmethod
    def get_original_volume_basename(stack, prep_id=None, detector_id=None, downscale=32, structure=None, volume_type='score', **kwargs):

        components = []
        if prep_id is not None:
            components.append('prep%(prep)d' % {'prep':prep_id})
        if detector_id is not None:
            components.append('detector%(detector_id)d' % {'detector_id':detector_id})
        if downscale is not None:
            components.append('down%(downscale)d' % {'downscale':downscale})
        tmp_str = '_'.join(components)
        basename = '%(stack)s_%(tmp_str)s_%(volstr)s' % \
            {'stack':stack, 'tmp_str':tmp_str, 'volstr':volume_type_to_str(volume_type)}
        if structure is not None:
            basename += '_' + structure
        return basename

    @staticmethod
    def get_warped_volume_basename(stack_m,
                                   stack_f=None,
                                   warp_setting=None,
                                   prep_id_m=None,
                                   prep_id_f=None,
                                   detector_id_m=None,
                                   detector_id_f=None,
                                   downscale=32,
                                   structure_m=None,
                                   structure_f=None,
                                   vol_type_m='score',
                                   vol_type_f='score',
                                   trial_idx=None,
                                   **kwargs):

        basename_m = DataManager.get_original_volume_basename(stack=stack_m, prep_id=prep_id_m, detector_id=detector_id_m,
                                                  downscale=downscale, volume_type=vol_type_m, structure=structure_m)
        
        if stack_f is None:
            assert warp_setting is None
            vol_name = basename_m
        else:
            basename_f = DataManager.get_original_volume_basename(stack=stack_f, prep_id=prep_id_f, detector_id=detector_id_f,
                                                  downscale=downscale, volume_type=vol_type_f, structure=structure_f)
            vol_name = basename_m + '_warp%(warp)d_' % {'warp':warp_setting} + basename_f

        if trial_idx is not None:
            vol_name += '_trial_%d' % trial_idx

        return vol_name

    @staticmethod
    def get_alignment_parameters_filepath(stack_f, stack_m,
                                          warp_setting,
                                          prep_id_m=None, prep_id_f=None,
                                          detector_id_m=None, detector_id_f=None,
                                          structure_f=None, structure_m=None,
                                          vol_type_f='score', vol_type_m='score', 
                                          downscale=32, 
                                          trial_idx=None):
        basename = DataManager.get_warped_volume_basename(**locals())
        fp = os.path.join(REGISTRATION_PARAMETERS_ROOTDIR, '%(stack_m)s',
                              '%(basename)s',
                              '%(basename)s_parameters.txt') % {'stack_m': stack_m, 'basename':basename}
        return fp

    @staticmethod
    def load_alignment_parameters(stack_f, stack_m, warp_setting,
                                  prep_id_m=None, prep_id_f=None,
                                  detector_id_m=None, detector_id_f=None,
                                  structure_f=None, structure_m=None,
                                  vol_type_f='score', vol_type_m='score',
                                  downscale=32, trial_idx=None):
        params_fp = DataManager.get_alignment_parameters_filepath(**locals())
        download_from_s3(params_fp, redownload=True)
        return DataManager.load_data(params_fp, 'transform_params')

    @staticmethod
    def save_alignment_parameters(fp, params, centroid_m, centroid_f, xdim_m, ydim_m, zdim_m, xdim_f, ydim_f, zdim_f):

        create_if_not_exists(os.path.dirname(fp))
        with open(fp, 'w') as f:
            f.write(array_to_one_liner(params))
            f.write(array_to_one_liner(centroid_m))
            f.write(array_to_one_liner([xdim_m, ydim_m, zdim_m]))
            f.write(array_to_one_liner(centroid_f))
            f.write(array_to_one_liner([xdim_f, ydim_f, zdim_f]))

    @staticmethod
    def get_alignment_result_filepath(stack_f, stack_m, warp_setting, what, ext=None,
                                      detector_id_m=None, detector_id_f=None,
                                      prep_id_m=None, prep_id_f=None,
                                      structure_f=None, structure_m=None, 
                                      vol_type_f='score', vol_type_m='score',
                                      downscale=32, trial_idx=None):
        reg_basename = DataManager.get_warped_volume_basename(**locals())
        if what == 'parameters':
            ext = 'txt'
        elif what == 'scoreHistory' or what == 'trajectory':
            ext = 'bp'
        elif what == 'scoreEvolution':
            ext = 'png'
        elif what == 'parametersWeightedAverage':
            ext = 'pkl'
        fp = os.path.join(REGISTRATION_PARAMETERS_ROOTDIR, stack_m, reg_basename, reg_basename + '_' + what + '.' + ext)
        return fp

    ####### Best trial index file #########

    @staticmethod
    def get_best_trial_index_filepath(stack_f, stack_m, warp_setting,
    classifier_setting_m=None, classifier_setting_f=None,
    type_f='score', type_m='score', downscale=32, param_suffix=None):
        basename = DataManager.get_warped_volume_basename(**locals())
        if param_suffix is None:
            fp = os.path.join(REGISTRATION_PARAMETERS_ROOTDIR, stack_m, basename + '_bestTrial', basename + '_bestTrial.txt')
        else:
            fp = os.path.join(REGISTRATION_PARAMETERS_ROOTDIR, stack_m, basename + '_bestTrial', basename + '_bestTrial_%(param_suffix)s.txt' % \
                             {'param_suffix':param_suffix})
        return fp

    @staticmethod
    def load_best_trial_index(stack_f, stack_m, warp_setting,
    classifier_setting_m=None, classifier_setting_f=None,
    type_f='score', type_m='score', downscale=32, param_suffix=None):
        fp = DataManager.get_best_trial_index_filepath(**locals())
        download_from_s3(fp)
        with open(fp, 'r') as f:
            best_trial_index = int(f.readline())
        return best_trial_index

    @staticmethod
    def load_best_trial_index_all_structures(stack_f, stack_m, warp_setting,
    classifier_setting_m=None, classifier_setting_f=None,
    type_f='score', type_m='score', downscale=32):
        input_kwargs = locals()
        best_trials = {}
        for structure in all_known_structures_sided:
            try:
                best_trials[structure] = DataManager.load_best_trial_index(param_suffix=structure, **input_kwargs)
            except Exception as e:
                sys.stderr.write(str(e) + '\n')
                sys.stderr.write("Best trial file for structure %s is not found.\n" % structure)
        return best_trials


    @staticmethod
    def get_alignment_viz_filepath(stack_m, stack_f,                                   
                                   warp_setting,
                                    section,
                                   prep_id_m=None, prep_id_f=None,
                                   detector_id_m=None, detector_id_f=None,
                                    vol_type_m='score', vol_type_f='score',
                                    downscale=32,
                                    trial_idx=None,
                                  out_downscale=32):
        """
        Args:
            downscale (int): downscale of both volumes (must be consistent).
            out_downsample (int): downscale of the output visualization images.
        """

        reg_basename = DataManager.get_warped_volume_basename(**locals())
        return os.path.join(REGISTRATION_VIZ_ROOTDIR, stack_m, reg_basename, 'down'+str(out_downscale), reg_basename + '_%04d_down%d.jpg' % (section, out_downscale))

    @staticmethod
    def load_confidence(stack_m, stack_f,
                            classifier_setting_m, classifier_setting_f, warp_setting, what,
                            type_m='score', type_f='score', param_suffix=None,
                            trial_idx=None):
        fp = DataManager.get_confidence_filepath(**locals())
        download_from_s3(fp)
        return load_pickle(fp)

    @staticmethod
    def get_confidence_filepath(stack_m, stack_f,
                            classifier_setting_m, classifier_setting_f, warp_setting, what,
                            type_m='score', type_f='score', param_suffix=None,
                            trial_idx=None):
        basename = DataManager.get_warped_volume_basename(**locals())

        if param_suffix is None:
            fn = basename + '_parameters' % {'param_suffix':param_suffix}
        else:
            fn = basename + '_parameters_%(param_suffix)s' % {'param_suffix':param_suffix}

        if what == 'hessians':
            return os.path.join(REGISTRATION_PARAMETERS_ROOTDIR, stack_m, basename + '_hessians', fn + '_hessians.pkl')
        #elif what == 'hessiansZscoreBased':
        #    return os.path.join(REGISTRATION_PARAMETERS_ROOTDIR, stack_m, basename + '_hessiansZscoreBased', fn + '_hessiansZscoreBased.pkl')
        elif what == 'zscores':
            return os.path.join(REGISTRATION_PARAMETERS_ROOTDIR, stack_m, basename + '_zscores', fn + '_zscores.pkl')
        elif what == 'score_landscape':
            return os.path.join(REGISTRATION_PARAMETERS_ROOTDIR, stack_m, basename + '_scoreLandscape', fn + '_scoreLandscape.png')
        elif what == 'score_landscape_rotations':
            return os.path.join(REGISTRATION_PARAMETERS_ROOTDIR, stack_m, basename + '_scoreLandscapeRotations', fn + '_scoreLandscapeRotations.png')
        elif what == 'peak_width':
            return os.path.join(REGISTRATION_PARAMETERS_ROOTDIR, stack_m, basename + '_peakWidth', fn + '_peakWidth.pkl')
        elif what == 'peak_radius':
            return os.path.join(REGISTRATION_PARAMETERS_ROOTDIR, stack_m, basename + '_peakRadius', fn + '_peakRadius.pkl')
        elif what == 'peak_radius_rotations':
            return os.path.join(REGISTRATION_PARAMETERS_ROOTDIR, stack_m, basename + '_peakRadiusRotations', fn + '_peakRadiusRotations.pkl')
        elif what == 'hessians_rotations':
            return os.path.join(REGISTRATION_PARAMETERS_ROOTDIR, stack_m, basename + '_hessiansRotations', fn + '_hessiansRotations.pkl')
        elif what == 'zscores_rotations':
            return os.path.join(REGISTRATION_PARAMETERS_ROOTDIR, stack_m, basename + '_zscoresRotations', fn + '_zscoresRotations.pkl')

        raise Exception("Unrecognized confidence type %s" % what)

    @staticmethod
    def get_classifier_filepath(structure, classifier_id):
        clf_fp = os.path.join(CLF_ROOTDIR, 'setting_%(setting)s', 'classifiers', '%(structure)s_clf_setting_%(setting)d.dump') % {'structure': structure, 'setting':classifier_id}
        return clf_fp

    @staticmethod
    def load_classifiers(classifier_id, structures=all_known_structures):

        from sklearn.externals import joblib

        clf_allClasses = {}
        for structure in structures:
            clf_fp = DataManager.get_classifier_filepath(structure=structure, classifier_id=classifier_id)
            download_from_s3(clf_fp)
            if os.path.exists(clf_fp):
                clf_allClasses[structure] = joblib.load(clf_fp)
            else:
                sys.stderr.write('Setting %d: No classifier found for %s.\n' % (classifier_id, structure))

        return clf_allClasses

#     @staticmethod
#     def load_sparse_scores(stack, structure, classifier_id, sec=None, fn=None, anchor_fn=None):

#         if fn is None:
#             fn = metadata_cache['sections_to_filenames'][stack][sec]

#         if anchor_fn is None:
#             anchor_fn = metadata_cache['anchor_fn'][stack]

#         sparse_scores_fn = DataManager.get_sparse_scores_filepath(stack=stack, structure=structure,
#                                             classifier_id=classifier_id, fn=fn, anchor_fn=anchor_fn)
#         download_from_s3(sparse_scores_fn)
#         return DataManager.load_data(sparse_scores_fn, filetype='bp')

    @staticmethod
    def load_sparse_scores(stack, structure, detector_id, prep_id=2, sec=None, fn=None):

        if fn is None:
            fn = metadata_cache['sections_to_filenames'][stack][sec]

        sparse_scores_fp = DataManager.get_sparse_scores_filepath(**locals())
        download_from_s3(sparse_scores_fp)
        return DataManager.load_data(sparse_scores_fp, filetype='bp')

    @staticmethod
    def get_sparse_scores_filepath(stack, structure, detector_id, prep_id=2, sec=None, fn=None):
        if fn is None:
            fn = metadata_cache['sections_to_filenames'][stack][sec]

        return os.path.join(SPARSE_SCORES_ROOTDIR, stack,
                            fn + '_prep%d'%prep_id,
                            'detector%d'%detector_id,
                            fn + '_prep%d'%prep_id + '_detector%d'%detector_id + '_' + structure + '_sparseScores.bp')


#     @staticmethod
#     def get_sparse_scores_filepath(stack, structure, classifier_id, sec=None, fn=None, anchor_fn=None):
#         if fn is None:
#             fn = metadata_cache['sections_to_filenames'][stack][sec]

#         if anchor_fn is None:
#             anchor_fn = metadata_cache['anchor_fn'][stack]

#         return os.path.join(SPARSE_SCORES_ROOTDIR, stack, '%(fn)s_lossless_alignedTo_%(anchor_fn)s_cropped', \
#                 '%(fn)s_lossless_alignedTo_%(anchor_fn)s_cropped_%(structure)s_sparseScores_setting_%(classifier_id)s.hdf') % \
#                 {'fn': fn, 'anchor_fn': anchor_fn, 'structure':structure, 'classifier_id': classifier_id}

    @staticmethod
    def load_intensity_volume(stack, downscale=32):
        fn = DataManager.get_intensity_volume_filepath(stack=stack, downscale=downscale)
        return DataManager.load_data(fn, filetype='bp')

    @staticmethod
    def get_intensity_volume_filepath(stack, downscale=32):
        basename = DataManager.get_original_volume_basename(stack=stack, volume_type='intensity', downscale=downscale)
        vol_fn = os.path.join(VOLUME_ROOTDIR, stack, basename, basename + '.bp')
        return vol_fn

    @staticmethod
    def get_intensity_volume_bbox_filepath(stack, downscale=32):
        basename = DataManager.get_original_volume_basename(volume_type='intensity', **locals())
        return os.path.join(VOLUME_ROOTDIR, stack, basename, basename + '_bbox.txt')

    @staticmethod
    def get_intensity_volume_metaimage_filepath(stack, downscale=32):
        """
        Returns:
            (header *.mhd filepath, data *.raw filepath)
        """
        basename = DataManager.get_original_volume_basename(stack=stack, volume_type='intensity_metaimage', downscale=downscale)
        vol_mhd_fp = os.path.join(VOLUME_ROOTDIR, stack, basename, basename + '.mhd')
        vol_raw_fp = os.path.join(VOLUME_ROOTDIR, stack, basename, basename + '.raw')
        return vol_mhd_fp, vol_raw_fp
    
    @staticmethod
    def get_intensity_volume_mask_metaimage_filepath(stack, downscale=32):
        """
        Returns:
            (header *.mhd filepath, data *.raw filepath)
        """
        basename = DataManager.get_original_volume_basename(stack=stack, volume_type='intensity_metaimage', downscale=downscale)
        vol_mhd_fp = os.path.join(VOLUME_ROOTDIR, stack, basename, basename + '_mask.mhd')
        vol_raw_fp = os.path.join(VOLUME_ROOTDIR, stack, basename, basename + '_mask.raw')
        return vol_mhd_fp, vol_raw_fp

    @staticmethod
    def load_annotation_as_score_volume(stack, downscale, structure):
        fn = DataManager.get_annotation_as_score_volume_filepath(**locals())
        return DataManager.load_data(fn, filetype='bp')

    @staticmethod
    def get_annotation_as_score_volume_filepath(stack, downscale, structure):
        basename = DataManager.get_original_volume_basename(volume_type='annotation_as_score', **locals())
        vol_fn = os.path.join(VOLUME_ROOTDIR, stack, basename, 'score_volumes', basename + '.bp')
        return vol_fn

    @staticmethod
    def load_annotation_volume(stack, downscale):
        fp = DataManager.get_annotation_volume_filepath(**locals())
        download_from_s3(fp)
        return DataManager.load_data(fp, filetype='bp')

    @staticmethod
    def get_annotation_volume_filepath(stack, downscale):
        basename = DataManager.get_original_volume_basename(volume_type='annotation', **locals())
        vol_fn = os.path.join(VOLUME_ROOTDIR, stack, basename, basename + '.bp')
        return vol_fn

    @staticmethod
    def get_annotation_volume_bbox_filepath(stack, downscale=32):
        basename = DataManager.get_original_volume_basename(volume_type='annotation', **locals())
        return os.path.join(VOLUME_ROOTDIR, stack, basename, basename + '_bbox.txt')

    @staticmethod
    def get_annotation_volume_label_to_name_filepath(stack):
        basename = DataManager.get_original_volume_basename(volume_type='annotation', **locals())
        fn = os.path.join(VOLUME_ROOTDIR, stack, basename, basename + '_nameToLabel.txt')
        # fn = os.path.join(volume_dir, stack, stack+'_down32_annotationVolume_nameToLabel.txt')
        return fn

    @staticmethod
    def load_annotation_volume_label_to_name(stack):
        fn = DataManager.get_annotation_volume_label_to_name_filepath(stack)
        download_from_s3(fn)
        label_to_name, name_to_label = DataManager.load_data(fn, filetype='label_name_map')
        return label_to_name, name_to_label

    ################
    # Mesh related #
    ################

    @staticmethod
    def load_shell_mesh(stack, downscale, return_polydata_only=True):
        shell_mesh_fn = DataManager.get_shell_mesh_filepath(stack, downscale)
        return load_mesh_stl(shell_mesh_fn, return_polydata_only=return_polydata_only)

    @staticmethod
    def get_shell_mesh_filepath(stack, downscale):
        basename = DataManager.get_original_volume_basename(stack=stack, downscale=downscale, volume_type='outer_contour')
        shell_mesh_fn = os.path.join(MESH_ROOTDIR, stack, basename, basename + "_smoothed.stl")
        return shell_mesh_fn

    
    @staticmethod
    def get_mesh_filepath(stack_m,
                            structure,
                            detector_id_m=None,
                            detector_id_f=None,
                            warp_setting=None,
                            stack_f=None,
                            downscale=32,
                            vol_type_m='score', 
                          vol_type_f='score',
                            trial_idx=None, **kwargs):
        basename = DataManager.get_warped_volume_basename(**locals())
        fn = basename + '_%s' % structure
        return os.path.join(MESH_ROOTDIR, stack_m, basename, fn + '.stl')

    
    @staticmethod
    def load_mesh(stack_m,
                                    structure,
                                    detector_id_m=None,
                                    stack_f=None,
                                    detector_id_f=None,
                                    warp_setting=None,
                                    downscale=32,
                                    vol_type_m='score', vol_type_f='score',
                                    trial_idx=None,
                                    return_polydata_only=True,
                                    **kwargs):
        mesh_fp = DataManager.get_mesh_filepath(**locals())
        mesh = load_mesh_stl(mesh_fp, return_polydata_only=return_polydata_only)
        if mesh is None:
            raise Exception('Mesh is empty: %s.' % structure)
        return mesh

    @staticmethod
    def load_meshes(stack_m,
                                    stack_f=None,
                                    detector_id_m=None,
                                    detector_id_f=None,
                                    warp_setting=None,
                                    downscale=32,
                                    vol_type_m='score', vol_type_f='score',
                                    trial_idx=None,
                                    structures=None,
                                    sided=True,
                                    return_polydata_only=True,
                                   include_surround=False):

        kwargs = locals()

        if structures is None:
            if sided:
                if include_surround:
                    structures = all_known_structures_sided_with_surround
                else:
                    structures = all_known_structures_sided
            else:
                structures = all_known_structures

        meshes = {}
        for structure in structures:
            try:
                meshes[structure] = DataManager.load_mesh(structure=structure, **kwargs)
            except Exception as e:
                sys.stderr.write('Error loading mesh for %s: %s.\n' % (structure, e))

        return meshes
    
    
#     @staticmethod
#     def load_atlas_mesh(atlas_name, structure, return_polydata_only=True, **kwargs):
#         mesh_fp = DataManager.get_structure_mean_mesh_filepath(atlas_name=atlas_name, structure=structure)
#         mesh = load_mesh_stl(mesh_fp, return_polydata_only=return_polydata_only)
#         if mesh is None:
#             raise Exception('Mesh is empty: %s.' % structure)
#         return mesh
    
#     @staticmethod
#     def load_atlas_meshes(atlas_name, structures=None, sided=True, return_polydata_only=True, include_surround=False):
#         kwargs = locals()
#         if structures is None:
#             if sided:
#                 if include_surround:
#                     structures = all_known_structures_sided_with_surround
#                 else:
#                     structures = all_known_structures_sided
#             else:
#                 structures = all_known_structures

#         meshes = {}
#         for structure in structures:
#             try:
#                 meshes[structure] = DataManager.load_atlas_mesh(atlas_name=atlas_name, structure=structure)
#             except Exception as e:
#                 sys.stderr.write('Error loading mesh for %s: %s.\n' % (structure, e))

#         return meshes
    
    @staticmethod
    def get_atlas_canonical_centroid_filepath(atlas_name, **kwargs):
        """
        Filepath of the atlas canonical centroid data. The centroid is with respect to aligned uncropped MD589.
        """
        return os.path.join(MESH_ROOTDIR, atlas_name, atlas_name + '_canonicalCentroid.txt')
    
    @staticmethod
    def get_atlas_canonical_normal_filepath(atlas_name, **kwargs):
        """
        Filepath of the atlas canonical centroid data.
        """
        return os.path.join(MESH_ROOTDIR, atlas_name, atlas_name + '_canonicalNormal.txt')
    
    @staticmethod
    def get_structure_mean_positions_filepath(atlas_name, **kwargs):
        """
        Filepath of the structure mean positions.        
        """
        return os.path.join(MESH_ROOTDIR, atlas_name, atlas_name + '_meanPositions.pkl')
        
    @staticmethod
    def get_instance_centroids_filepath(atlas_name, **kwargs):
        """
        Filepath of the structure mean positions.        
        """
        return os.path.join(MESH_ROOTDIR, atlas_name, atlas_name + '_instanceCentroids.pkl')
    
    @staticmethod
    def get_structure_viz_filepath(atlas_name, structure, suffix, **kwargs):
        """
        """
        return os.path.join(MESH_ROOTDIR, atlas_name, 'visualizations', structure, atlas_name + '_' + structure + '_' + suffix + '.png')
    
    @staticmethod
    def get_structure_mean_shape_filepath(atlas_name, structure, **kwargs):
        """
        """
        return os.path.join(MESH_ROOTDIR, atlas_name, 'mean_shapes', atlas_name + '_' + structure + '_meanShape.bp')

    @staticmethod
    def get_structure_mean_shape_origin_filepath(atlas_name, structure, **kwargs):
        """
        Mean shape origin, relative to the template instance's centroid.
        """
        return os.path.join(MESH_ROOTDIR, atlas_name, 'mean_shapes', atlas_name + '_' + structure + '_meanShapeOrigin.txt')
    
    @staticmethod
    def get_structure_mean_mesh_filepath(atlas_name, structure, **kwargs):
        """
        Structure mean mesh, relative to the template instance's centroid.
        """
        return os.path.join(MESH_ROOTDIR, atlas_name, 'mean_shapes', atlas_name + '_' + structure + '_meanMesh.stl')
    
    @staticmethod
    def get_instance_mesh_filepath(atlas_name, structure, index, **kwargs):
        """
        Filepath of the instance mesh to derive mean shapes in atlas.
        
        Args:
            index (int): the index of the instance. The template instance is at index 0.
        
        """
        return os.path.join(MESH_ROOTDIR, atlas_name, 'instance_meshes', atlas_name + '_' + structure + '_' + str(index) + '.stl')

    @staticmethod
    def get_instance_sources_filepath(atlas_name, structure, **kwargs):
        """
        Path to the instance mesh sources file.
        """
        return os.path.join(MESH_ROOTDIR, atlas_name, 'instance_sources', atlas_name + '_' + structure + '_sources.pkl')
    

    ###############
    # Volumes I/O #
    ###############
    
    @staticmethod
    def load_volume_all_known_structures(stack_m, stack_f,
                                        warp_setting,
                                        classifier_setting_m=None,
                                        classifier_setting_f=None,
                                        type_m='score',
                                        type_f='score',
                                        downscale=32,
                                        structures=None,
                                        trial_idx=None,
                                        sided=True,
                                        include_surround=False):
        if stack_f is not None:
            return DataManager.load_transformed_volume_all_known_structures(**locals())
        else:
            raise Exception('Not implemented.')


    @staticmethod
    def load_transformed_volume(stack_m, stack_f,
                                warp_setting,
                                detector_id_m=None,
                                detector_id_f=None,
                                prep_id_m=None,
                                prep_id_f=None,
                                structure_f=None,
                                structure_m=None,
                                vol_type_m='score',
                                vol_type_f='score',
                                structure=None,
                                downscale=32,
                                trial_idx=None):
        fp = DataManager.get_transformed_volume_filepath(**locals())
        download_from_s3(fp)
        return DataManager.load_data(fp, filetype='bp')


    @staticmethod
    def load_transformed_volume_all_known_structures(stack_m,
                                                     stack_f,
                                                    warp_setting,
                                                    detector_id_m=None,
                                                    detector_id_f=None,
                                                     prep_id_m=None,
                                                     prep_id_f=None,
                                                    vol_type_m='score',
                                                    vol_type_f='score',
                                                    downscale=32,
                                                    structures=None,
                                                    sided=True,
                                                    include_surround=False,
                                                     trial_idx=None,
                                                     return_label_mappings=False,
                                                     name_or_index_as_key='name'
):
        """
        Load transformed volumes for all structures.
        
        Args:
            trial_idx: could be int (for global transform) or dict {sided structure name: best trial index} (for local transform).
            
        Returns:
            if return_label_mappings is True, returns (volumes, structure_to_label, label_to_structure), volumes is dict.
            else, returns volumes.
        """

        if structures is None:
            if sided:
                if include_surround:
                    structures = all_known_structures_sided_with_surround
                else:
                    structures = all_known_structures_sided
            else:
                structures = all_known_structures

        loaded = False
        sys.stderr.write('Prior structure/index map not found. Generating a new one.\n')

        volumes = {}
        if not loaded:
            structure_to_label = {}
            label_to_structure = {}
            index = 1
        for structure in structures:
            try:

                if loaded:
                    index = structure_to_label[structure]

                if trial_idx is None or isinstance(trial_idx, int):
                    v = DataManager.load_transformed_volume(stack_m=stack_m, vol_type_m=vol_type_m,
                                                            stack_f=stack_f, vol_type_f=vol_type_f, 
                                                            downscale=downscale,
                                                            prep_id_m=prep_id_m,
                                                            prep_id_f=prep_id_f,
                                                            detector_id_m=detector_id_m,
                                                            detector_id_f=detector_id_f,
                                                            warp_setting=warp_setting,
                                                            structure=structure,
                                                            trial_idx=trial_idx)

                else:
                    v = DataManager.load_transformed_volume(stack_m=stack_m, vol_type_m=vol_type_m,
                                                            stack_f=stack_f, vol_type_f=vol_type_f, 
                                                            downscale=downscale,
                                                            prep_id_m=prep_id_m,
                                                            prep_id_f=prep_id_f,
                                                            detector_id_m=detector_id_m,
                                                            detector_id_f=detector_id_f,
                                                            warp_setting=warp_setting,
                                                            structure=structure,
                                                            trial_idx=trial_idx[convert_to_nonsurround_label(structure)])

                if name_or_index_as_key == 'name':
                    volumes[structure] = v
                else:
                    volumes[index] = v

                if not loaded:
                    structure_to_label[structure] = index
                    label_to_structure[index] = structure
                    index += 1

            except Exception as e:
                sys.stderr.write('%s\n' % e)
                sys.stderr.write('Score volume for %s does not exist.\n' % structure)

        if return_label_mappings:
            return volumes, structure_to_label, label_to_structure
        else:
            return volumes

    @staticmethod
    def get_transformed_volume_filepath(stack_m, stack_f,
                                        warp_setting,
                                        prep_id_m=None,
                                        prep_id_f=None,
                                        detector_id_m=None,
                                        detector_id_f=None,
                                        structure_m=None,
                                        structure_f=None,
                                        downscale=32,
                                        vol_type_m='score',
                                        vol_type_f='score',
                                        structure=None,
                                        trial_idx=None):

        basename = DataManager.get_warped_volume_basename(**locals())
        if structure is not None:
            fn = basename + '_%s' % structure
        else:
            fn = basename
        return os.path.join(VOLUME_ROOTDIR, stack_m, basename, 'score_volumes', fn + '.bp')
    
    @staticmethod
    def load_transformed_volume_bbox(stack_m, stack_f,
                                        warp_setting,
                                        prep_id_m=None,
                                        prep_id_f=None,
                                        detector_id_m=None,
                                        detector_id_f=None,
                                        structure_m=None,
                                        structure_f=None,
                                        downscale=32,
                                        vol_type_m='score',
                                        vol_type_f='score',
                                        structure=None,
                                        trial_idx=None):
        fp = DataManager.get_transformed_volume_bbox_filepath(**locals())
        download_from_s3(fp)
        return np.loadtxt(fp)
    
    @staticmethod
    def get_transformed_volume_bbox_filepath(stack_m, stack_f,
                                        warp_setting,
                                        prep_id_m=None,
                                        prep_id_f=None,
                                        detector_id_m=None,
                                        detector_id_f=None,
                                        structure_m=None,
                                        structure_f=None,
                                        downscale=32,
                                        vol_type_m='score',
                                        vol_type_f='score',
                                        structure=None,
                                        trial_idx=None):

        basename = DataManager.get_warped_volume_basename(**locals())
        if structure is not None:
            fn = basename + '_%s' % structure
        else:
            fn = basename
        return os.path.join(VOLUME_ROOTDIR, stack_m, basename, 'score_volumes', fn + '_bbox.txt')

    

    ##########################
    ## Probabilistic Shape  ##
    ##########################

    @staticmethod
    def load_prob_shapes(stack_m, stack_f=None,
            classifier_setting_m=None,
            classifier_setting_f=None,
            warp_setting=None,
            downscale=32,
            type_m='score', type_f='score',
            trial_idx=0,
            structures=None,
            sided=True,
            return_polydata_only=True):

        kwargs = locals()

        if structures is None:
            if sided:
                structures = all_known_structures_sided
            else:
                structures = all_known_structures

        prob_shapes = {}
        for structure in structures:
            try:
                vol_fp = DataManager.get_prob_shape_volume_filepath(structure=structure, **kwargs)
                download_from_s3(vol_fp)
                vol = bp.unpack_ndarray_file(vol_fp)

                origin_fp = DataManager.get_prob_shape_origin_filepath(structure=structure, **kwargs)
                download_from_s3(origin_fp)
                origin = np.loadtxt(origin_fp)

                prob_shapes[structure] = (vol, origin)
            except Exception as e:
                sys.stderr.write('Error loading probablistic shape for %s: %s\n' % (structure, e))

        return prob_shapes

#     @staticmethod
#     def get_prob_shape_viz_filepath(stack_m, structure,
#                                     stack_f=None,
#                                         warp_setting=None,
#                                         classifier_setting_m=None,
#                                         classifier_setting_f=None,
#                                         downscale=32,
#                                          type_m='score',
#                                          type_f='score',
#                                         trial_idx=0,
#                                         suffix=None,
#                                         **kwargs):
#         """
#         Return prob. shape volume filepath.
#         """

#         basename = DataManager.get_warped_volume_basename(**locals())
#         assert structure is not None
#         fn = basename + '_' + structure + '_' + suffix
#         return os.path.join(, stack_m, basename, 'probabilistic_shape_viz', structure, fn + '.png')

#     @staticmethod
#     def get_prob_shape_volume_filepath(stack_m, stack_f=None,
#                                         warp_setting=None,
#                                         classifier_setting_m=None,
#                                         classifier_setting_f=None,
#                                         downscale=32,
#                                          type_m='score',
#                                          type_f='score',
#                                         structure=None,
#                                         trial_idx=0, **kwargs):
#         """
#         Return prob. shape volume filepath.
#         """

#         basename = DataManager.get_warped_volume_basename(**locals())
#         if structure is not None:
#             fn = basename + '_' + structure

#         return os.path.join(VOLUME_ROOTDIR, stack_m, basename, 'probabilistic_shapes', fn + '.bp')

#     @staticmethod
#     def get_prob_shape_origin_filepath(stack_m, stack_f=None,
#                                         warp_setting=None,
#                                         classifier_setting_m=None,
#                                         classifier_setting_f=None,
#                                         downscale=32,
#                                          type_m='score',
#                                          type_f='score',
#                                         structure=None,
#                                         trial_idx=0, **kwargs):
#         """
#         Return prob. shape volume origin filepath.

#         Note that these origins are with respect to

#         """

#         basename = DataManager.get_warped_volume_basename(**locals())
#         if structure is not None:
#             fn = basename + '_' + structure
#         return os.path.join(VOLUME_ROOTDIR, stack_m, basename, 'probabilistic_shapes', fn + '_origin.txt')

    @staticmethod
    def get_volume_filepath(stack_m, stack_f=None,
                                        warp_setting=None,
                                        classifier_setting_m=None,
                                        classifier_setting_f=None,
                                        downscale=32,
                                         type_m='score',
                                          type_f='score',
                                        structure=None,
                                        trial_idx=None):

        basename = DataManager.get_warped_volume_basename(**locals())

        if structure is not None:
            fn = basename + '_' + structure

        if type_m == 'score':
            return DataManager.get_score_volume_filepath(stack=stack_m, structure=structure, downscale=downscale)
        else:
            raise

    @staticmethod
    def get_score_volume_filepath(stack, structure, volume_type='score', prep_id=None, detector_id=None, downscale=32):
        basename = DataManager.get_original_volume_basename(stack=stack, detector_id=detector_id, prep_id=prep_id, volume_type=volume_type)
        vol_fp = os.path.join(VOLUME_ROOTDIR, '%(stack)s',
                              '%(basename)s',
                              'score_volumes',
                             '%(basename)s_%(struct)s.bp') % \
        {'stack':stack, 'basename':basename, 'struct':structure}
        return vol_fp

    @staticmethod
    def get_score_volume_bbox_filepath(stack, structure, downscale, detector_id, prep_id=2, volume_type='score'):
        basename = DataManager.get_original_volume_basename(stack=stack, detector_id=detector_id, prep_id=prep_id, volume_type=volume_type)
        fp = os.path.join(VOLUME_ROOTDIR, '%(stack)s',
                              '%(basename)s',
                              'score_volumes',
                             '%(basename)s_%(struct)s_bbox.txt') % \
        {'stack':stack, 'basename':basename, 'struct':structure}
        return fp

    @staticmethod
    def get_volume_gradient_filepath_template(stack, structure, prep_id=None, detector_id=None, downscale=32, volume_type='score', **kwargs):
        basename = DataManager.get_original_volume_basename(stack=stack, prep_id=prep_id, detector_id=detector_id, downscale=downscale, volume_type=volume_type, **kwargs)
        grad_fp = os.path.join(VOLUME_ROOTDIR, '%(stack)s',
                              '%(basename)s',
                              'score_volume_gradients',
                             '%(basename)s_%(struct)s_%%(suffix)s.bp') % \
        {'stack':stack, 'basename':basename, 'struct':structure}
        return grad_fp

    @staticmethod
    def get_volume_gradient_filepath(stack, structure, suffix, volume_type='score', prep_id=None, detector_id=None, downscale=32):
        grad_fp = DataManager.get_volume_gradient_filepath_template(**locals())  % {'suffix': suffix}
        return grad_fp

    @staticmethod
    def load_original_volume_all_known_structures(stack, downscale=32, detector_id=None, prep_id=None, structures=None, sided=True, volume_type='score', return_structure_index_mapping=True, include_surround=False):
        """
        Args:
            return_structure_index_mapping (bool): if True, return both volumes and structure-label mapping. If False, return only volumes.

        Returns:
        """

        if structures is None:
            if sided:
                if include_surround:
                    structures = all_known_structures_sided_with_surround
                else:
                    structures = all_known_structures_sided
            else:
                structures = all_known_structures

        if return_structure_index_mapping:

            try:
                label_to_structure, structure_to_label = DataManager.load_volume_label_to_name(stack=stack)
                loaded = True
                sys.stderr.write('Load structure/index map.\n')
            except:
                loaded = False
                sys.stderr.write('Prior structure/index map not found. Generating a new one.\n')

            volumes = {}
            if not loaded:
                structure_to_label = {}
                label_to_structure = {}
                index = 1
            for structure in sorted(structures):
                try:
                    if loaded:
                        index = structure_to_label[structure]

                    volumes[index] = DataManager.load_original_volume(stack=stack, structure=structure,
                                            downscale=downscale, detector_id=detector_id, prep_id=prep_id,
                                            volume_type=volume_type)
                    if not loaded:
                        structure_to_label[structure] = index
                        label_to_structure[index] = structure
                        index += 1
                except Exception as e:
                    sys.stderr.write('Score volume for %s does not exist: %s\n' % (structure, e))

            # One volume at down=32 takes about 1MB of memory.

            sys.stderr.write('Volume shape: (%d, %d, %d)\n' % volumes.values()[0].shape)
            return volumes, structure_to_label, label_to_structure

        else:
            volumes = {}
            for structure in structures:
                try:
                    volumes[structure] = DataManager.load_original_volume(stack=stack, structure=structure,
                                            downscale=downscale, detector_id=detector_id, prep_id=prep_id,
                                            volume_type=volume_type)
                except:
                    sys.stderr.write('Score volume for %s does not exist.\n' % structure)

            sys.stderr.write('Volume shape: (%d, %d, %d)\n' % volumes.values()[0].shape)
            return volumes


    @staticmethod
    def get_original_volume_filepath(stack, structure, prep_id=None, detector_id=None, volume_type='score', downscale=32):
        if volume_type == 'score':
            fp = DataManager.get_score_volume_filepath(**locals())
        elif volume_type == 'annotation':
            fp = DataManager.get_annotation_volume_filepath(stack=stack, downscale=downscale)
        elif volume_type == 'annotationAsScore':
            fp = DataManager.get_score_volume_filepath(**locals())
        elif volume_type == 'intensity':
            fp = DataManager.get_intensity_volume_filepath(stack=stack, downscale=downscale)
        elif volume_type == 'intensity_mhd':
            fp = DataManager.get_intensity_volume_mhd_filepath(stack=stack, downscale=downscale)
        else:
            raise Exception("Volume type must be one of score, annotation, annotationAsScore or intensity.")
        return fp

    @staticmethod
    def load_original_volume(stack, structure, downscale, prep_id=None, detector_id=None, volume_type='score'):
        vol_fp = DataManager.get_original_volume_filepath(**locals())
        download_from_s3(vol_fp, is_dir=False)
        volume = DataManager.load_data(vol_fp, filetype='bp')
        return volume

    @staticmethod
    def load_original_volume_bbox(stack, volume_type, prep_id=None, detector_id=None, structure=None, downscale=32):
        """
        This returns the 3D bounding box.

        Args:
            volume_type (str):
                annotation: with respect to aligned uncropped thumbnail
                score/thumbnail: with respect to aligned cropped thumbnail
                shell: with respect to aligned uncropped thumbnail
        """
        bbox_fp = DataManager.get_original_volume_bbox_filepath(**locals())
        download_from_s3(bbox_fp)
        volume_bbox = DataManager.load_data(bbox_fp, filetype='bbox')
        return volume_bbox


    @staticmethod
    def get_original_volume_bbox_filepath(stack,
                                detector_id=None,
                                          prep_id=None,
                                downscale=32,
                                 volume_type='score',
                                structure=None):
        if volume_type == 'annotation':
            bbox_fn = DataManager.get_annotation_volume_bbox_filepath(stack=stack)
        elif volume_type == 'score':
            bbox_fn = DataManager.get_score_volume_bbox_filepath(**locals())
        elif volume_type == 'annotationAsScore':
            bbox_fn = DataManager.get_score_volume_bbox_filepath(**locals())
        elif volume_type == 'shell':
            bbox_fn = DataManager.get_shell_bbox_filepath(stack, structure, downscale)
        elif volume_type == 'thumbnail':
            bbox_fn = DataManager.get_score_volume_bbox_filepath(stack=stack, structure='7N', downscale=downscale,
            detector_id=detector_id)
        else:
            raise Exception('Type must be annotation, score, shell or thumbnail.')

        return bbox_fn

    @staticmethod
    def get_shell_bbox_filepath(stack, label, downscale):
        bbox_filepath = VOLUME_ROOTDIR + '/%(stack)s/%(stack)s_down%(ds)d_outerContourVolume_bbox.txt' % \
                        dict(stack=stack, ds=downscale)
        return bbox_filepath


    #########################
    ###     Score map     ###
    #########################

    @staticmethod
    def get_image_version_str(stack, version, resolution='lossless', downscale=None, anchor_fn=None):

        if resolution == 'thumbnail':
            downscale = 32

        if anchor_fn is None:
            anchor_fn = metadata_cache['anchor_fn'][stack]

        basename = resolution + '_alignedTo_' + anchor_fn + '_' + version + '_down' + str(downscale)
        return basename

    @staticmethod
    def get_scoremap_viz_filepath(stack, downscale, detector_id, prep_id=2, section=None, fn=None, structure=None):

        if section is not None:
            fn = metadata_cache['sections_to_filenames'][stack][section]
            if is_invalid(fn):
                raise Exception('Section is invalid: %s.' % fn)

        scoremap_viz_fp = os.path.join(SCOREMAP_VIZ_ROOTDIR, 'down%(smdown)d',
                                       '%(struct)s', '%(stack)s', 'detector%(detector_id)d',
                                       'prep%(prep)s', '%(fn)s_prep%(prep)d_down%(smdown)d_%(struct)s_detector%(detector_id)s_scoremapViz.jpg') % {'stack':stack, 'struct':structure, 'smdown':downscale, 'prep':prep_id, 'fn':fn, 'detector_id':detector_id}

        return scoremap_viz_fp

    @staticmethod
    def get_downscaled_scoremap_filepath(stack, structure, detector_id, downscale, prep_id=2, section=None, fn=None):

        if section is not None:
            fn = metadata_cache['sections_to_filenames'][stack][section]
            if is_invalid(fn):
                raise Exception('Section is invalid: %s.' % fn)

        scoremap_bp_filepath = os.path.join(SCOREMAP_ROOTDIR, 'down%(smdown)d',
                                            '%(stack)s',
                                            '%(stack)s_prep%(prep)d_down%(smdown)d_detector%(detector_id)d',
                                           '%(fn)s_prep%(prep)d_down%(smdown)d_detector%(detector_id)d',
                                           '%(fn)s_prep%(prep)d_down%(smdown)d_detector%(detector_id)d_%(structure)s_scoremap.bp') % {'stack':stack, 'prep':prep_id, 'fn': fn, 'smdown':downscale, 'detector_id': detector_id, 'structure':structure}

        return scoremap_bp_filepath

    @staticmethod
    def load_downscaled_scoremap(stack, structure, detector_id, downscale, prep_id=2, section=None, fn=None):
        """
        Return scoremap as bp file.
        """

        scoremap_bp_filepath = DataManager.get_downscaled_scoremap_filepath(**locals())
        download_from_s3(scoremap_bp_filepath)

        if not os.path.exists(scoremap_bp_filepath):
            raise Exception('No scoremap for image %s (section %d) for label %s\n' % \
            (metadata_cache['sections_to_filenames'][stack][section], section, structure))

        scoremap_downscaled = DataManager.load_data(scoremap_bp_filepath, filetype='bp')
        return scoremap_downscaled

    @staticmethod
    def load_scoremap(stack, structure, detector_id, downscale, prep_id=2, section=None, fn=None):
        """
        Return scoremap as bp file.
        """
        return DataManager.load_downscaled_scoremap(**locals())


#     @staticmethod
#     def load_downscaled_scoremap(stack, structure, classifier_id, section=None, fn=None, anchor_fn=None, downscale=32):
#         """
#         Return scoremaps as bp files.
#         """

#         # Load scoremap
#         scoremap_bp_filepath = DataManager.get_downscaled_scoremap_filepath(stack, section=section, \
#                         fn=fn, anchor_fn=anchor_fn, structure=structure, classifier_id=classifier_id,
#                         downscale=downscale)

#         download_from_s3(scoremap_bp_filepath)

#         if not os.path.exists(scoremap_bp_filepath):
#             raise Exception('No scoremap for image %s (section %d) for label %s\n' % \
#             (metadata_cache['sections_to_filenames'][stack][section], section, structure))

#         scoremap_downscaled = DataManager.load_data(scoremap_bp_filepath, filetype='bp')
#         return scoremap_downscaled

#     @staticmethod
#     def get_scoremap_filepath(stack, structure, classifier_id, section=None, fn=None, anchor_fn=None, return_bbox_fp=False):

#         if section is not None:
#             fn = metadata_cache['sections_to_filenames'][stack][section]
#             if is_invalid(fn):
#                 raise Exception('Section is invalid: %s.' % fn)

#         if anchor_fn is None:
#             anchor_fn = metadata_cache['anchor_fn'][stack]

#         scoremap_bp_filepath = os.path.join(SCOREMAPS_ROOTDIR, stack, \
#         '%(fn)s_lossless_alignedTo_%(anchor_fn)s_cropped/%(fn)s_lossless_alignedTo_%(anchor_fn)s_cropped_%(structure)s_denseScoreMap_setting_%(classifier_id)d.hdf') \
#         % dict(stack=stack, fn=fn, structure=structure, anchor_fn=anchor_fn, classifier_id=classifier_id)

#         scoremap_bbox_filepath = os.path.join(SCOREMAPS_ROOTDIR, stack, \
#         '%(fn)s_lossless_alignedTo_%(anchor_fn)s_cropped/%(fn)s_lossless_alignedTo_%(anchor_fn)s_cropped_%(structure)s_denseScoreMap_interpBox.txt') \
#             % dict(stack=stack, fn=fn, structure=structure, anchor_fn=anchor_fn)

#         if return_bbox_fp:
#             return scoremap_bp_filepath, scoremap_bbox_filepath
#         else:
#             return scoremap_bp_filepath

#     @staticmethod
#     def load_scoremap(stack, structure, classifier_id, section=None, fn=None, anchor_fn=None, downscale=1):
#         """
#         Return scoremaps.
#         """

#         # Load scoremap
#         scoremap_bp_filepath, scoremap_bbox_filepath = DataManager.get_scoremap_filepath(stack, section=section, \
#                                     fn=fn, anchor_fn=anchor_fn, structure=structure, return_bbox_fp=True, classifier_id=classifier_id)
#         if not os.path.exists(scoremap_bp_filepath):
#             raise Exception('No scoremap for image %s (section %d) for label %s\n' % \
#             (metadata_cache['sections_to_filenames'][stack][section], section, structure))

#         scoremap = DataManager.load_data(scoremap_bp_filepath, filetype='hdf')

#         # Load interpolation box
#         xmin, xmax, ymin, ymax = DataManager.load_data(scoremap_bbox_filepath, filetype='bbox')
#         ymin_downscaled = ymin / downscale
#         xmin_downscaled = xmin / downscale

#         full_width, full_height = metadata_cache['image_shape'][stack]
#         scoremap_downscaled = np.zeros((full_height/downscale, full_width/downscale), np.float32)

#         # To conserve memory, it is important to make a copy of the sub-scoremap and delete the original scoremap
#         scoremap_roi_downscaled = scoremap[::downscale, ::downscale].copy()
#         del scoremap

#         h_downscaled, w_downscaled = scoremap_roi_downscaled.shape

#         scoremap_downscaled[ymin_downscaled : ymin_downscaled + h_downscaled,
#                             xmin_downscaled : xmin_downscaled + w_downscaled] = scoremap_roi_downscaled

#         return scoremap_downscaled

    ###########################
    ######  CNN Features ######
    ###########################

    # @staticmethod
    # def load_dnn_feature_locations(stack, model_name, section=None, fn=None, anchor_fn=None, input_img_version='cropped_gray'):
    #     fp = DataManager.get_dnn_feature_locations_filepath(stack=stack, model_name=model_name, section=section, fn=fn, anchor_fn=anchor_fn, input_img_version=input_img_version)
    #     download_from_s3(fp)
    #     locs = np.loadtxt(fp).astype(np.int)
    #     indices = locs[:, 0]
    #     locations = locs[:, 1:]
    #     return indices, locations

    # @staticmethod
    # def load_dnn_feature_locations(stack, model_name, section=None, fn=None, prep_id=2, win=1, input_img_version='gray'):
    #     fp = DataManager.get_dnn_feature_locations_filepath(stack=stack, model_name=model_name, section=section, fn=fn, prep_id=prep_id, input_img_version=input_img_version, win=win)
    #     download_from_s3(fp)
    #     locs = np.loadtxt(fp).astype(np.int)
    #     indices = locs[:, 0]
    #     locations = locs[:, 1:]
    #     return indices, locations

    @staticmethod
    def load_patch_locations(stack, win, section=None, fn=None, prep_id=2, input_img_version='gray'):
        fp = DataManager.get_patch_locations_filepath(**locals())
        download_from_s3(fp)
        locs = np.loadtxt(fp).astype(np.int)
        indices = locs[:, 0]
        locations = locs[:, 1:]
        return indices, locations

#     @staticmethod
#     def get_dnn_feature_locations_filepath(stack, model_name, section=None, fn=None, anchor_fn=None, input_img_version='cropped_gray'):

#         if fn is None:
#             fn = metadata_cache['sections_to_filenames'][stack][section]

#         if anchor_fn is None:
#             anchor_fn = metadata_cache['anchor_fn'][stack]

#         image_version_basename = DataManager.get_image_version_basename(stack=stack, resol='lossless', version=input_img_version)
#         image_basename = DataManager.get_image_basename(stack=stack, fn=fn, resol='lossless', version=input_img_version)

#         # feature_locs_fn = os.path.join(PATCH_FEATURES_ROOTDIR, model_name, stack, \
#         # '%(fn)s_lossless_alignedTo_%(anchor_fn)s_cropped/%(fn)s_lossless_alignedTo_%(anchor_fn)s_cropped_patch_locations.txt' % \
#         # dict(fn=fn, anchor_fn=anchor_fn))

#         feature_locs_fn = os.path.join(PATCH_FEATURES_ROOTDIR, model_name, stack, image_version_basename,
#                                        image_basename + '_patch_locations.txt')
#         return feature_locs_fn

#     @staticmethod
#     def get_dnn_feature_locations_filepath(stack, model_name, section=None, fn=None, prep_id=2, input_img_version='gray', win=1):

#         if fn is None:
#             fn = metadata_cache['sections_to_filenames'][stack][section]
#         feature_locs_fp = os.path.join(PATCH_FEATURES_ROOTDIR, model_name, stack,
#                                        stack+'_prep%(prep)d'%{'prep':prep_id}+'_'+input_img_version+'_win%(win)d'%{'win':win},
#                                        fn+'_prep%(prep)d'%{'prep':prep_id}+'_'+input_img_version+'_win%(win)d'%{'win':win}+'_'+model_name+'_patch_locations.txt')
#         return feature_locs_fp

#     @staticmethod
#     def get_dnn_feature_locations_filepath_v2(stack, section=None, fn=None, prep_id=2, input_img_version='gray', win=1):

#         if fn is None:
#             fn = metadata_cache['sections_to_filenames'][stack][section]
#         feature_locs_fp = os.path.join(PATCH_FEATURES_ROOTDIR, stack,
#                                        stack+'_prep%(prep)d'%{'prep':prep_id}+'_'+input_img_version+'_win%(win)d'%{'win':win},
#                                        fn+'_prep%(prep)d'%{'prep':prep_id}+'_'+input_img_version+'_win%(win)d'%{'win':win}+'_patchLocations.txt')
#         return feature_locs_fp

    @staticmethod
    def get_patch_locations_filepath(stack, win, section=None, fn=None, prep_id=2, input_img_version='gray'):

        if fn is None:
            fn = metadata_cache['sections_to_filenames'][stack][section]
        feature_locs_fp = os.path.join(PATCH_LOCATIONS_ROOTDIR, stack,
                                       stack+'_prep%(prep)d'%{'prep':prep_id}+'_'+input_img_version+'_win%(win)d'%{'win':win},
                                       fn+'_prep%(prep)d'%{'prep':prep_id}+'_'+input_img_version+'_win%(win)d'%{'win':win}+'_patchLocations.txt')
        return feature_locs_fp

#     @staticmethod
#     def get_patch_locations_filepath_v2(stack, win, section=None, fn=None, prep_id=2, input_img_version='gray'):

#         if fn is None:
#             fn = metadata_cache['sections_to_filenames'][stack][section]
#         feature_locs_fp = os.path.join(PATCH_LOCATIONS_ROOTDIR, stack,
#                                        stack+'_prep%(prep)d'%{'prep':prep_id}+'_'+input_img_version+'_win%(win)d'%{'win':win},
#                                        fn+'_prep%(prep)d'%{'prep':prep_id}+'_'+input_img_version+'_win%(win)d'%{'win':win}+'_patchLocations.txt')
#         return feature_locs_fp

#     @staticmethod
#     def get_dnn_features_filepath(stack, model_name, section=None, fn=None, anchor_fn=None, input_img_version='cropped_gray'):
#         """
#         Args:
#             version (str): default is cropped_gray.
#         """

#         if fn is None:
#             fn = metadata_cache['sections_to_filenames'][stack][section]

#         if anchor_fn is None:
#             anchor_fn = metadata_cache['anchor_fn'][stack]

#         image_version_basename = DataManager.get_image_version_basename(stack=stack, resol='lossless', version=input_img_version)
#         image_basename = DataManager.get_image_basename(stack=stack, fn=fn, resol='lossless', version=input_img_version)

#         feature_fn = os.path.join(PATCH_FEATURES_ROOTDIR, model_name, stack, image_version_basename, image_basename + '_features.bp')

#         return feature_fn

    @staticmethod
    def get_dnn_features_filepath(stack, model_name, win, section=None, fn=None, prep_id=2, input_img_version='gray', suffix=None):
        """
        Args:
            version (str): default is cropped_gray.
        """
        if fn is None:
            fn = metadata_cache['sections_to_filenames'][stack][section]

        if suffix is None:
            feature_fp = os.path.join(PATCH_FEATURES_ROOTDIR, model_name, stack,
                                       stack+'_prep%(prep)d'%{'prep':prep_id}+'_'+input_img_version+'_win%(win)d'%{'win':win},
                                       fn+'_prep%(prep)d'%{'prep':prep_id}+'_'+input_img_version+'_win%(win)d'%{'win':win}+'_'+model_name+'_features.bp')
        else:
            feature_fp = os.path.join(PATCH_FEATURES_ROOTDIR, model_name, stack,
                                       stack+'_prep%(prep)d'%{'prep':prep_id}+'_'+input_img_version+'_win%(win)d'%{'win':win},
                                       fn+'_prep%(prep)d'%{'prep':prep_id}+'_'+input_img_version+'_win%(win)d'%{'win':win}+'_'+model_name+'_features_' + suffix + '.bp')

        return feature_fp

    @staticmethod
    def load_dnn_features(stack, model_name, win, section=None, fn=None, input_img_version='gray', prep_id=2, suffix=None):
        """
        Args:
            version (str): default is cropped_gray.
        """

        features_fp = DataManager.get_dnn_features_filepath(**locals())
        download_from_s3(features_fp)

        try:
            return load_hdf(features_fp)
        except:
            pass

        try:
            return load_hdf_v2(features_fp)
        except:
            pass

        return bp.unpack_ndarray_file(features_fp)

#     @staticmethod
#     def load_dnn_features(stack, model_name, section=None, fn=None, anchor_fn=None, input_img_version='cropped_gray'):
#         """
#         Args:
#             version (str): default is cropped_gray.
#         """

#         features_fp = DataManager.get_dnn_features_filepath(stack=stack, model_name=model_name, section=section, fn=fn, anchor_fn=anchor_fn, input_img_version=input_img_version)
#         download_from_s3(features_fp)

#         try:
#             return load_hdf(features_fp)
#         except:
#             pass

#         try:
#             return load_hdf_v2(features_fp)
#         except:
#             pass

#         return bp.unpack_ndarray_file(features_fp)

    ##################
    ##### Image ######
    ##################

    @staticmethod
    def get_image_version_basename(stack, version, resol='lossless', anchor_fn=None):

        if anchor_fn is None:
            anchor_fn = metadata_cache['anchor_fn'][stack]

        if resol == 'lossless' and (version == 'cropped' or version == 'cropped_tif'):
            image_version_basename = stack + '_' + resol + '_alignedTo_' + anchor_fn + '_cropped'
        else:
            image_version_basename = stack + '_' + resol + '_alignedTo_' + anchor_fn + '_' + version

        return image_version_basename

    @staticmethod
    def get_image_basename(stack, version, resol='lossless', anchor_fn=None, fn=None, section=None):

        if anchor_fn is None:
            anchor_fn = metadata_cache['anchor_fn'][stack]

        if section is not None:
            fn = metadata_cache['sections_to_filenames'][stack][section]
            assert is_invalid(fn=fn), 'Section is invalid: %s.' % fn

        if resol == 'lossless' and (version == 'cropped' or version == 'cropped_tif'):
            image_basename = fn + '_' + resol + '_alignedTo_' + anchor_fn + '_cropped'
        else:
            image_basename = fn + '_' + resol + '_alignedTo_' + anchor_fn + '_' + version

        return image_basename


#     @staticmethod
#     def get_image_basename_v2(stack, version, resol='lossless', anchor_fn=None, fn=None, section=None):

#         if anchor_fn is None:
#             anchor_fn = metadata_cache['anchor_fn'][stack]

#         if section is not None:
#             fn = metadata_cache['sections_to_filenames'][stack][section]
#             assert is_invalid(fn=fn), 'Section is invalid: %s.' % fn

#         if resol == 'lossless' and (version == 'cropped' or version == 'cropped_tif'):
#             image_basename = fn + '_' + resol + '_alignedTo_' + anchor_fn + '_cropped'
#         else:
#             image_basename = fn + '_' + resol + '_alignedTo_' + anchor_fn + '_' + version

#         return image_basename

    @staticmethod
    def get_image_dir_v2(stack, prep_id, version=None, resol='lossless',
                      data_dir=DATA_DIR, raw_data_dir=RAW_DATA_DIR, thumbnail_data_dir=THUMBNAIL_DATA_DIR):
        """
        Args:
            version (str): version string
            data_dir: This by default is DATA_DIR, but one can change this ad-hoc when calling the function

        Returns:
            Absolute path of the image directory.
        """

        if version is None:
            if resol == 'thumbnail':
                image_dir = os.path.join(thumbnail_data_dir, stack, stack + '_prep%d' % prep_id + '_%s' % resol)
            else:
                image_dir = os.path.join(data_dir, stack, stack + '_prep%d' % prep_id + '_%s' % resol)
        else:
            if resol == 'thumbnail':
                image_dir = os.path.join(thumbnail_data_dir, stack, stack + '_prep%d' % prep_id + '_%s' % resol + '_' + version)
            else:
                image_dir = os.path.join(data_dir, stack, stack + '_prep%d' % prep_id + '_%s' % resol + '_' + version)

        return image_dir


    @staticmethod
    def get_image_dir(stack, version, resol='lossless', anchor_fn=None, modality=None,
                      data_dir=DATA_DIR, raw_data_dir=RAW_DATA_DIR, thumbnail_data_dir=THUMBNAIL_DATA_DIR):
        """
        Args:
            data_dir: This by default is DATA_DIR, but one can change this ad-hoc when calling the function
            resol: can be either lossless or thumbnail
            version: TODO - Write a list of options
            modality: can be either nissl or fluorescent. If not specified, it is inferred.

        Returns:
            Absolute path of the image directory.
        """

        if anchor_fn is None:
            anchor_fn = DataManager.load_anchor_filename(stack)

        if resol == 'lossless' and version == 'original_jp2':
            image_dir = os.path.join(raw_data_dir, stack)
        elif resol == 'lossless' and version == 'jpeg':
            assert modality == 'nissl'
            image_dir = os.path.join(data_dir, stack, stack+'_'+resol+'_alignedTo_%(anchor_fn)s_cropped_compressed' % {'anchor_fn':anchor_fn})
        elif resol == 'lossless' and version == 'uncropped_tif':
            image_dir = os.path.join(data_dir, stack, stack + '_' + resol + '_tif')
        elif resol == 'lossless' and version == 'cropped_16bit':
            image_dir = os.path.join(data_dir, stack, stack+'_'+resol+'_alignedTo_%(anchor_fn)s_cropped' % {'anchor_fn':anchor_fn})
        elif resol == 'thumbnail' and (version == 'cropped' or version == 'cropped_tif'):
            image_dir = os.path.join(thumbnail_data_dir, stack, stack+'_'+resol+'_alignedTo_%(anchor_fn)s_cropped' % {'anchor_fn':anchor_fn})
        elif (resol == 'thumbnail' and version == 'aligned') or (resol == 'thumbnail' and version == 'aligned_tif'):
            image_dir = os.path.join(thumbnail_data_dir, stack, stack+'_'+resol+'_alignedTo_%(anchor_fn)s' % {'anchor_fn':anchor_fn})
        elif resol == 'thumbnail' and version == 'original_png':
            image_dir = os.path.join(raw_data_dir, stack)
        else:
            # sys.stderr.write('No special rule for (%s, %s). So using the default image directory composition rule.\n' % (version, resol))
            image_dir = os.path.join(data_dir, stack, stack + '_' + resol + '_alignedTo_' + anchor_fn + '_' + version)

        return image_dir

    @staticmethod
    def load_image_v2(stack, prep_id, resol='lossless', version=None, section=None, fn=None, data_dir=DATA_DIR, ext=None, thumbnail_data_dir=THUMBNAIL_DATA_DIR):
        img_fp = DataManager.get_image_filepath_v2(**locals())
        if resol == 'lossless':
            download_from_s3(img_fp, local_root=DATA_ROOTDIR)
        else:
            download_from_s3(img_fp, local_root=THUMBNAIL_DATA_ROOTDIR)
        return imread(img_fp)

    @staticmethod
    def load_image(stack, version, resol='lossless', section=None, fn=None, anchor_fn=None, modality=None, data_dir=DATA_DIR, ext=None):
        img_fp = DataManager.get_image_filepath(**locals())
        download_from_s3(img_fp)
        return imread(img_fp)

    @staticmethod
    def get_image_filepath_v2(stack, prep_id, version=None, resol='lossless',
                           data_dir=DATA_DIR, raw_data_dir=RAW_DATA_DIR, thumbnail_data_dir=THUMBNAIL_DATA_DIR,
                           section=None, fn=None, ext=None):
        """
        Args:
        	version (str): the version string.

        Returns:
            Absolute path of the image file.
        """

        if section is not None:
            fn = metadata_cache['sections_to_filenames'][stack][section]
            if is_invalid(fn=fn):
                raise Exception('Section is invalid: %s.' % fn)
        else:
            assert fn is not None

        image_dir = DataManager.get_image_dir_v2(stack=stack, prep_id=prep_id, resol=resol, version=version, data_dir=data_dir, thumbnail_data_dir=thumbnail_data_dir)
        if ext is None:
            if version == 'mask':
                ext = 'png'
            elif version == 'contrastStretched' or version == 'grayJpeg' or version == 'jpeg' or version == 'grayDefaultJpeg':
                ext = 'jpg'
            else:
                ext = 'tif'

        if version is None:
            image_name = fn + '_prep%d' % prep_id + '_%s' % resol + '.' + ext
        else:
            image_name = fn + '_prep%d' % prep_id + '_' + resol + '_' + version + '.' + ext
        image_path = os.path.join(image_dir, image_name)

        return image_path

    @staticmethod
    def get_image_filepath(stack, version, resol='lossless',
                           data_dir=DATA_DIR, raw_data_dir=RAW_DATA_DIR, thumbnail_data_dir=THUMBNAIL_DATA_DIR,
                           section=None, fn=None, anchor_fn=None, modality=None, ext=None):
        """
        Args:
            data_dir: This by default is DATA_DIR, but one can change this ad-hoc when calling the function
            resol: can be either lossless or thumbnail
            version: TODO - Write a list of options
            modality: can be either nissl or fluorescent. If not specified, it is inferred.

        Returns:
            Absolute path of the image file.
        """

        image_name = None

        if section is not None:
            fn = metadata_cache['sections_to_filenames'][stack][section]
            if is_invalid(fn=fn):
                raise Exception('Section is invalid: %s.' % fn)
        else:
            assert fn is not None

        if anchor_fn is None:
            anchor_fn = DataManager.load_anchor_filename(stack)

        if modality is None:
            if (stack in all_alt_nissl_ntb_stacks or stack in all_alt_nissl_tracing_stacks) and fn.split('-')[1][0] == 'F':
                modality = 'fluorescent'
            else:
                modality = 'nissl'

        image_dir = DataManager.get_image_dir(stack=stack, version=version, resol=resol, modality=modality, data_dir=data_dir)

        if resol == 'thumbnail' and version == 'original_png':
            image_name = fn + '.png'
        elif resol == 'thumbnail' and (version == 'cropped' or version == 'cropped_tif'):
            image_name = '_'.join([fn, resol, 'alignedTo_' + anchor_fn + '_cropped.tif'])
        elif resol == 'thumbnail' and (version == 'aligned' or version == 'aligned_tif'):
            image_name = '_'.join([fn, resol, 'alignedTo_' + anchor_fn + '.tif'])
        elif resol == 'lossless' and version == 'jpeg':
            assert modality == 'nissl'
            image_name = '_'.join([fn, resol, 'alignedTo_%(anchor_fn)s_cropped_compressed.jpg' % {'anchor_fn':anchor_fn}])
        elif resol == 'lossless' and version == 'cropped':
            image_name = '_'.join([fn, resol, 'alignedTo_%(anchor_fn)s_cropped.tif' % {'anchor_fn':anchor_fn}])
        elif resol == 'lossless' and version == 'uncropped_tif':
            image_name = fn + '_lossless.tif'
        elif resol == 'lossless' and version == 'cropped_gray_jpeg':
            image_name = fn + '_' + resol + '_alignedTo_' + anchor_fn + '_cropped_gray.jpg'
        else:
            if ext is None:
                ext = 'tif'
            image_name = '_'.join([fn, resol, 'alignedTo_' + anchor_fn + '_' + version + '.' + ext])

        image_path = os.path.join(image_dir, image_name)

        return image_path


    @staticmethod
    def get_image_dimension(stack):
        """
        Returns:
            (image width, image height).
        """

        first_sec, last_sec = DataManager.load_cropbox(stack)[4:]
        # anchor_fn = DataManager.load_anchor_filename(stack)
        # filename_to_section, section_to_filename = DataManager.load_sorted_filenames(stack)

        # for i in range(10, 13):
        random_fn = metadata_cache['valid_filenames'][stack][0]
        # random_fn = section_to_filename[i]
        # fp = DataManager.get_image_filepath(stack=stack, resol='thumbnail', version='cropped', fn=random_fn, anchor_fn=anchor_fn)
        # try:
        img = DataManager.load_image_v2(stack=stack, resol='thumbnail', prep_id=2, fn=random_fn)
            # break
        # except:
        #     pass

        image_height, image_width = img.shape[:2]
        image_height = image_height * 32
        image_width = image_width * 32

        return image_width, image_height

    #######################################################

    @staticmethod
    def convert_section_to_z(stack, sec, downsample, z_begin=None, first_sec=None):
        """
        Because the z-spacing is much larger than pixel size on x-y plane,
        the theoretical voxels are square on x-y plane but elongated in z-direction.
        In practice we need to represent volume using cubic voxels.
        This function computes the z-coordinate for a given section number.
        This depends on the downsample factor of the volume.

        Args:
            downsample (int):
            first_sec (int): default to the first brainstem section defined in ``cropbox".
            z_begin (float): default to the z position of the first_sec.

        Returns:
            z1, z2 (2-tuple of float): in
        """

        xy_pixel_distance = XY_PIXEL_DISTANCE_LOSSLESS * downsample
        voxel_z_size = SECTION_THICKNESS / xy_pixel_distance
        # Voxel size in z direction in unit of x,y pixel.

        if first_sec is None:
            first_sec, _ = DataManager.load_cropbox(stack)[4:]

        if z_begin is None:
            z_begin = first_sec * voxel_z_size

        z1 = sec * voxel_z_size
        z2 = (sec + 1) * voxel_z_size
        return z1-z_begin, z2-1-z_begin

    @staticmethod
    def convert_z_to_section(stack, z, downsample, z_begin=None):
        """
        z_begin default to int(np.floor(first_sec*voxel_z_size)).
        """
        xy_pixel_distance = XY_PIXEL_DISTANCE_LOSSLESS * downsample
        voxel_z_size = SECTION_THICKNESS / xy_pixel_distance
        # print 'voxel size:', xy_pixel_distance, xy_pixel_distance, voxel_z_size, 'um'

        # first_sec, last_sec = section_range_lookup[stack]
        first_sec, last_sec = DataManager.load_cropbox(stack)[4:]
        # z_end = int(np.ceil((last_sec+1)*voxel_z_size))

        if z_begin is None:
            # z_begin = int(np.floor(first_sec*voxel_z_size))
            z_begin = first_sec * voxel_z_size
        # print 'z_begin', first_sec*voxel_z_size, z_begin

        sec_float = np.float32((z + z_begin) / voxel_z_size) # if use np.float, will result in np.floor(98.0)=97
        # print sec_float
        # print sec_float == 98., np.floor(np.float(sec_float))
        sec_floor = int(np.floor(sec_float))

        return sec_floor


    @staticmethod
    def get_initial_snake_contours_filepath(stack):
        return os.path.join(THUMBNAIL_DATA_DIR, stack, stack + '_prep1_thumbnail_initSnakeContours.pkl')

    @staticmethod
    def get_anchor_initial_snake_contours_filepath(stack):
        return os.path.join(THUMBNAIL_DATA_DIR, stack, stack + '_prep1_thumbnail_anchorInitSnakeContours.pkl')

    # @staticmethod
    # def get_auto_submask_rootdir_filepath(stack):
    #     """
    #     Args:
    #         what (str): submask or decisions.
    #         submask_ind (int): if what is submask, must provide submask_ind.
    #     """
    #     anchor_fn = metadata_cache['anchor_fn'][stack]
    #     dir_path = os.path.join(THUMBNAIL_DATA_DIR, stack, stack + '_alignedTo_' + anchor_fn + '_auto_submasks')
    #     return dir_path

    @staticmethod
    def get_auto_submask_rootdir_filepath(stack):
        return os.path.join(THUMBNAIL_DATA_DIR, stack, stack + '_prep1_thumbnail_autoSubmasks')

    @staticmethod
    def get_auto_submask_dir_filepath(stack, fn=None, sec=None):
        submasks_dir = DataManager.get_auto_submask_rootdir_filepath(stack)
        if fn is None:
            fn = metadata_cache['sections_to_filenames'][stack][sec]
        dir_path = os.path.join(submasks_dir, fn)
        return dir_path

#     @staticmethod
#     def get_auto_submask_filepath(stack, what, submask_ind=None, fn=None, sec=None):
#         """
#         Args:
#             what (str): submask or decisions.
#             submask_ind (int): if what is submask, must provide submask_ind.
#         """
#         anchor_fn = metadata_cache['anchor_fn'][stack]
#         dir_path = DataManager.get_auto_submask_dir_filepath(stack=stack, fn=fn, sec=sec)

#         if what == 'submask':
#             assert submask_ind is not None, "Must provide submask_ind."
#             fp = os.path.join(dir_path, fn + '_alignedTo_' + anchor_fn + '_auto_submask_%d.png' % submask_ind)
#         elif what == 'decisions':
#             fp = os.path.join(dir_path, fn + '_alignedTo_' + anchor_fn + '_auto_submask_decisions.csv')
#         else:
#             raise Exception("Not recognized.")

#         return fp

    @staticmethod
    def get_auto_submask_filepath(stack, what, submask_ind=None, fn=None, sec=None):
        """
        Args:
            what (str): submask or decisions.
            submask_ind (int): if what is submask, must provide submask_ind.
        """
        dir_path = DataManager.get_auto_submask_dir_filepath(stack=stack, fn=fn, sec=sec)

        if what == 'submask':
            assert submask_ind is not None, "Must provide submask_ind."
            fp = os.path.join(dir_path, fn + '_prep1_thumbnail_autoSubmask_%d.png' % submask_ind)
        elif what == 'decisions':
            fp = os.path.join(dir_path, fn + '_prep1_thumbnail_autoSubmaskDecisions.csv')
        else:
            raise Exception("Input %s is not recognized." % what)

        return fp

    @staticmethod
    def get_user_modified_submask_rootdir_filepath(stack):
        dir_path = os.path.join(THUMBNAIL_DATA_DIR, stack, stack + '_prep1_thumbnail_userModifiedSubmasks')
        return dir_path

    # @staticmethod
    # def get_user_modified_submask_rootdir_filepath(stack):
    #     """
    #     Args:
    #         what (str): submask or decisions.
    #         submask_ind (int): if what is submask, must provide submask_ind.
    #     """
    #     anchor_fn = metadata_cache['anchor_fn'][stack]
    #     dir_path = os.path.join(THUMBNAIL_DATA_DIR, stack, stack + '_alignedTo_' + anchor_fn + '_userModified_submasks')
    #     return dir_path

    # @staticmethod
    # def get_user_modified_submask_rootdir_filepath(stack):
    #     """
    #     Args:
    #         what (str): submask or decisions.
    #         submask_ind (int): if what is submask, must provide submask_ind.
    #     """
    #     anchor_fn = metadata_cache['anchor_fn'][stack]
    #     dir_path = os.path.join(THUMBNAIL_DATA_DIR, stack, stack + '_alignedTo_' + anchor_fn + '_userModified_submasks')
    #     return dir_path

    @staticmethod
    def get_user_modified_submask_dir_filepath(stack, fn=None, sec=None):
        submasks_dir = DataManager.get_user_modified_submask_rootdir_filepath(stack)
        if fn is None:
            fn = metadata_cache['sections_to_filenames'][stack][sec]
        dir_path = os.path.join(submasks_dir, fn)
        return dir_path

    @staticmethod
    def get_user_modified_submask_filepath(stack, what, submask_ind=None, fn=None, sec=None):
        """
        Args:
            what (str): submask or decisions.
            submask_ind (int): if what is submask, must provide submask_ind.
        """
        dir_path = DataManager.get_user_modified_submask_dir_filepath(stack=stack, fn=fn, sec=sec)

        if what == 'submask':
            assert submask_ind is not None, "Must provide submask_ind."
            fp = os.path.join(dir_path, fn + '_prep1_thumbnail_userModifiedSubmask_%d.png' % submask_ind)
        elif what == 'decisions':
            fp = os.path.join(dir_path, fn + '_prep1_thumbnail_userModifiedSubmaskDecisions.csv')
        elif what == 'parameters':
            fp = os.path.join(dir_path, fn + '_prep1_thumbnail_userModifiedParameters.json')
        elif what == 'contour_vertices':
            fp = os.path.join(dir_path, fn + '_prep1_thumbnail_userModifiedSubmaskContourVertices.pkl')
        else:
            raise Exception("Input %s is not recognized." % what)

        return fp


#     @staticmethod
#     def get_user_modified_submask_filepath(stack, what, submask_ind=None, fn=None, sec=None):
#         """
#         Args:
#             what (str): submask or decisions.
#             submask_ind (int): if what is submask, must provide submask_ind.
#         """
#         anchor_fn = metadata_cache['anchor_fn'][stack]
#         dir_path = DataManager.get_user_modified_submask_dir_filepath(stack=stack, fn=fn, sec=sec)

#         if what == 'submask':
#             assert submask_ind is not None, "Must provide submask_ind."
#             fp = os.path.join(dir_path, fn + '_alignedTo_' + anchor_fn + '_userModified_submask_%d.png' % submask_ind)
#         elif what == 'decisions':
#             fp = os.path.join(dir_path, fn + '_alignedTo_' + anchor_fn + '_userModified_submask_decisions.csv')
#         elif what == 'parameters':
#             fp = os.path.join(dir_path, fn + '_alignedTo_' + anchor_fn + '_userModified_parameters.json')
#         elif what == 'contour_vertices':
#             fp = os.path.join(dir_path, fn + '_alignedTo_' + anchor_fn + '_userModified_submask_contour_vertices.pkl')
#         else:
#             raise Exception("Not recognized.")

#         return fp


    # @staticmethod
    # def get_thumbnail_mask_dir_v3(stack, version='aligned'):
    #     """
    #     Get directory path of thumbnail mask.
    #
    #     Args:
    #         version (str): One of aligned, aligned_cropped, cropped.
    #     """
    #
    #     anchor_fn = metadata_cache['anchor_fn'][stack]
    #     if version == 'aligned':
    #         dir_path = os.path.join(THUMBNAIL_DATA_DIR, stack, stack + '_alignedTo_' + anchor_fn + '_masks')
    #     elif version == 'aligned_cropped' or version == 'cropped':
    #         dir_path = os.path.join(THUMBNAIL_DATA_DIR, stack, stack + '_alignedTo_' + anchor_fn + '_masks_cropped')
    #     else:
    #         raise Exception('version %s is not recognized.' % version)
    #     return dir_path

#     @staticmethod
#     def get_user_modified_submask_filepath(stack, what, submask_ind=None, fn=None, sec=None):
#         """
#         Args:
#             what (str): submask or decisions.
#             submask_ind (int): if what is submask, must provide submask_ind.
#         """
#         anchor_fn = metadata_cache['anchor_fn'][stack]
#         dir_path = DataManager.get_user_modified_submask_dir_filepath(stack=stack, fn=fn, sec=sec)

#         if what == 'submask':
#             assert submask_ind is not None, "Must provide submask_ind."
#             fp = os.path.join(dir_path, fn + '_alignedTo_' + anchor_fn + '_userModified_submask_%d.png' % submask_ind)
#         elif what == 'decisions':
#             fp = os.path.join(dir_path, fn + '_alignedTo_' + anchor_fn + '_userModified_submask_decisions.csv')
#         elif what == 'parameters':
#             fp = os.path.join(dir_path, fn + '_alignedTo_' + anchor_fn + '_userModified_parameters.json')
#         elif what == 'contour_vertices':
#             fp = os.path.join(dir_path, fn + '_alignedTo_' + anchor_fn + '_userModified_submask_contour_vertices.pkl')
#         else:
#             raise Exception("Not recognized.")

#         return fp


    # @staticmethod
    # def get_thumbnail_mask_dir_v3(stack, version='aligned'):
    #     """
    #     Get directory path of thumbnail mask.
    #
    #     Args:
    #         version (str): One of aligned, aligned_cropped, cropped.
    #     """
    #
    #     anchor_fn = metadata_cache['anchor_fn'][stack]
    #     if version == 'aligned':
    #         dir_path = os.path.join(THUMBNAIL_DATA_DIR, stack, stack + '_alignedTo_' + anchor_fn + '_masks')
    #     elif version == 'aligned_cropped' or version == 'cropped':
    #         dir_path = os.path.join(THUMBNAIL_DATA_DIR, stack, stack + '_alignedTo_' + anchor_fn + '_masks_cropped')
    #     else:
    #         raise Exception('version %s is not recognized.' % version)
    #     return dir_path

    @staticmethod
    def get_thumbnail_mask_dir_v3(stack, prep_id):
        """
        Get directory path of thumbnail mask.
        """
        return os.path.join(THUMBNAIL_DATA_DIR, stack, stack + '_prep%d_thumbnail_mask' % prep_id)

    @staticmethod
    def get_thumbnail_mask_filename_v3(stack, prep_id, section=None, fn=None):
        """
        Get filepath of thumbnail mask.
        """

        dir_path = DataManager.get_thumbnail_mask_dir_v3(stack, prep_id=prep_id)
        if fn is None:
            fn = metadata_cache['sections_to_filenames'][stack][section]
        if version == 'aligned':
            fp = os.path.join(dir_path, fn + '_masks_alignedTo_' + anchor_fn + '_mask.png')
        elif version == 'aligned_cropped' or version == 'cropped':
            fp = os.path.join(dir_path, fn + '_masks_alignedTo_' + anchor_fn + '_mask_cropped.png')
        else:
            raise Exception('version %s is not recognized.' % version)
        return fp

    # @staticmethod
    # def get_thumbnail_mask_filename_v3(stack, section=None, fn=None, version='aligned_cropped'):
    #     """
    #     Get filepath of thumbnail mask.
    #
    #     Args:
    #         version (str): One of aligned, aligned_cropped, cropped.
    #     """
    #
    #     anchor_fn = metadata_cache['anchor_fn'][stack]
    #     dir_path = DataManager.get_thumbnail_mask_dir_v3(stack, version=version)
    #     if fn is None:
    #         fn = metadata_cache['sections_to_filenames'][stack][section]
    #
    #     if version == 'aligned':
    #         fp = os.path.join(dir_path, fn + '_alignedTo_' + anchor_fn + '_mask.png')
    #     elif version == 'aligned_cropped' or version == 'cropped':
    #         fp = os.path.join(dir_path, fn + '_alignedTo_' + anchor_fn + '_mask_cropped.png')
    #     else:
    #         raise Exception('version %s is not recognized.' % version)
    #     return fp

    @staticmethod
    def load_thumbnail_mask_v3(stack, prep_id, section=None, fn=None):
        fp = DataManager.get_thumbnail_mask_filename_v3(stack=stack, section=section, fn=fn, prep_id=prep_id)
        download_from_s3(fp, local_root=DATA_ROOTDIR)
        mask = imread(fp).astype(np.bool)
        return mask

    # @staticmethod
    # def load_thumbnail_mask_v3(stack, prep_id, section=None, fn=None):
    #     if stack in ['MD589', 'MD585', 'MD594']:
    #         fp = DataManager.get_thumbnail_mask_filename_v2(stack=stack, section=section, fn=fn, prep_id=prep_id)
    #     else:
    #         fp = DataManager.get_thumbnail_mask_filename_v3(stack=stack, section=section, fn=fn, prep_id=prep_id)
    #     download_from_s3(fp)
    #     mask = imread(fp).astype(np.bool)
    #     return mask

    # @staticmethod
    # def load_thumbnail_mask_v3(stack, section=None, fn=None, version='aligned_cropped'):
    #     if stack in ['MD589', 'MD585', 'MD594']:
    #         fp = DataManager.get_thumbnail_mask_filename_v2(stack=stack, section=section, fn=fn, version=version)
    #     else:
    #         fp = DataManager.get_thumbnail_mask_filename_v3(stack=stack, section=section, fn=fn, version=version)
    #     download_from_s3(fp)
    #     mask = imread(fp).astype(np.bool)
    #     return mask

    # @staticmethod
    # def load_thumbnail_mask_v2(stack, section=None, fn=None, version='aligned_cropped'):
    #     fp = DataManager.get_thumbnail_mask_filename_v2(stack=stack, section=section, fn=fn, version=version)
    #     download_from_s3(fp, local_root=DATA_ROOTDIR)
    #     mask = DataManager.load_data(fp, filetype='image').astype(np.bool)
    #     return mask
    #
    # @staticmethod
    # def get_thumbnail_mask_dir_v2(stack, version='aligned_cropped'):
    #     anchor_fn = metadata_cache['anchor_fn'][stack]
    #     if version == 'aligned_cropped':
    #         mask_dir = os.path.join(THUMBNAIL_DATA_DIR, stack, stack + '_masks_alignedTo_' + anchor_fn + '_cropped')
    #     elif version == 'aligned':
    #         mask_dir = os.path.join(THUMBNAIL_DATA_DIR, stack, stack + '_masks_alignedTo_' + anchor_fn)
    #     else:
    #         raise Exception("version %s not recognized." % version)
    #     return mask_dir
    #
    # @staticmethod
    # def get_thumbnail_mask_filename_v2(stack, section=None, fn=None, version='aligned_cropped'):
    #     anchor_fn = metadata_cache['anchor_fn'][stack]
    #     sections_to_filenames = metadata_cache['sections_to_filenames'][stack]
    #     if fn is None:
    #         fn = sections_to_filenames[section]
    #     mask_dir = DataManager.get_thumbnail_mask_dir_v2(stack=stack, version=version)
    #     if version == 'aligned_cropped':
    #         fp = os.path.join(mask_dir, fn + '_mask_alignedTo_' + anchor_fn + '_cropped.png')
    #     elif version == 'aligned':
    #         fp = os.path.join(mask_dir, fn + '_mask_alignedTo_' + anchor_fn + '.png')
    #     else:
    #         raise Exception("version %s not recognized." % version)
    #     return fp

    ###################################

    @staticmethod
    def get_region_labels_filepath(stack, sec=None, fn=None):
        """
        Returns:
            dict {label: list of region indices}
        """
        if fn is None:
            fn = metadata_cache['sections_to_filenames'][stack][sec]
        return os.path.join(CELL_FEATURES_CLF_ROOTDIR, 'region_indices_by_label', stack, fn + '_region_indices_by_label.hdf')

    @staticmethod
    def get_ntb_to_nissl_intensity_profile_mapping_filepath(stack=None, ntb_fn=None):
        """
        Args:
            stack (str): If None, read the a priori mapping.
        """
        if stack is None:
            fp = os.path.join(DATA_DIR, 'average_nissl_intensity_mapping.npy')
        else:
            fp = os.path.join(DATA_DIR, stack, stack + '_intensity_mapping', '%s_intensity_mapping.npy' % (ntb_fn))

        return fp

    @staticmethod
    def get_dataset_dir(dataset_id):
        return os.path.join(CLF_ROOTDIR, 'datasets', 'dataset_%d' % dataset_id)
    
    @staticmethod
    def get_dataset_patches_filepath(dataset_id, structure=None):
        if structure is None:
            patch_images_fp = os.path.join(CLF_ROOTDIR, 'datasets', 'dataset_%d' % dataset_id, 'patch_images.hdf')
        else:
            patch_images_fp = os.path.join(CLF_ROOTDIR, 'datasets', 'dataset_%d' % dataset_id, 'patch_images_%s.hdf' % structure)
        return patch_images_fp

    @staticmethod
    def get_dataset_features_filepath(dataset_id, structure=None, ext='bp'):
        if structure is None:
            features_fp = os.path.join(CLF_ROOTDIR, 'datasets', 'dataset_%d' % dataset_id, 'patch_features.' + ext)
        else:
            features_fp = os.path.join(CLF_ROOTDIR, 'datasets', 'dataset_%d' % dataset_id, 'patch_features_%s.' % structure + ext)
        return features_fp

    @staticmethod
    def get_dataset_addresses_filepath(dataset_id, structure=None):
        if structure is None:
            addresses_fp = os.path.join(CLF_ROOTDIR, 'datasets', 'dataset_%d' % dataset_id, 'patch_addresses.pkl')
        else:
            addresses_fp = os.path.join(CLF_ROOTDIR, 'datasets', 'dataset_%d' % dataset_id, 'patch_addresses_%s.pkl' % structure)
        return addresses_fp

    @staticmethod
    def get_classifier_filepath(classifier_id, structure):
        classifier_id_dir = os.path.join(CLF_ROOTDIR, 'setting_%d' % classifier_id)
        classifier_dir = os.path.join(classifier_id_dir, 'classifiers')
        return os.path.join(classifier_dir, '%(structure)s_clf_setting_%(setting)d.dump' % \
                     dict(structure=structure, setting=classifier_id))

    ####### Fluorescent ########

    @staticmethod
    def get_labeled_neurons_filepath(stack, sec=None, fn=None):
        if fn is None:
            fn = metadata_cache['sections_to_filenames'][stack][sec]
        return os.path.join(LABELED_NEURONS_ROOTDIR, stack, fn, fn + ".pkl")


    @staticmethod
    def load_labeled_neurons_filepath(stack, sec=None, fn=None):
        fp = DataManager.get_labeled_neurons_filepath(**locals())
        download_from_s3(fp)
        return load_pickle(fp)

##################################################

def download_all_metadata():

    for stack in all_stacks:
        try:
            download_from_s3(DataManager.get_sorted_filenames_filename(stack=stack))
        except:
            pass
        try:
            download_from_s3(DataManager.get_anchor_filename_filename(stack=stack))
        except:
            pass
        try:
            download_from_s3(DataManager.get_cropbox_filename(stack=stack))
        except:
            pass

download_all_metadata()

# This module stores any meta information that is dynamic.
metadata_cache = {}

def generate_metadata_cache():

    global metadata_cache
    metadata_cache['image_shape'] = {}
    metadata_cache['anchor_fn'] = {}
    metadata_cache['sections_to_filenames'] = {}
    metadata_cache['filenames_to_sections'] = {}
    metadata_cache['section_limits'] = {}
    metadata_cache['cropbox'] = {}
    metadata_cache['valid_sections'] = {}
    metadata_cache['valid_filenames'] = {}
    metadata_cache['valid_sections_all'] = {}
    metadata_cache['valid_filenames_all'] = {}
    for stack in all_stacks:

        try:
            metadata_cache['anchor_fn'][stack] = DataManager.load_anchor_filename(stack)
        except:
            pass
        try:
            metadata_cache['sections_to_filenames'][stack] = DataManager.load_sorted_filenames(stack)[1]
        except:
            pass
        try:
            metadata_cache['filenames_to_sections'][stack] = DataManager.load_sorted_filenames(stack)[0]
            metadata_cache['filenames_to_sections'][stack].pop('Placeholder')
            metadata_cache['filenames_to_sections'][stack].pop('Nonexisting')
            metadata_cache['filenames_to_sections'][stack].pop('Rescan')
        except:
            pass
        try:
            metadata_cache['section_limits'][stack] = DataManager.load_cropbox(stack)[4:]
        except:
            pass
        try:
            metadata_cache['cropbox'][stack] = DataManager.load_cropbox(stack)[:4]
        except:
            pass

        try:
            first_sec, last_sec = metadata_cache['section_limits'][stack]
            metadata_cache['valid_sections'][stack] = [sec for sec in range(first_sec, last_sec+1) if not is_invalid(stack=stack, sec=sec)]
            metadata_cache['valid_filenames'][stack] = [metadata_cache['sections_to_filenames'][stack][sec] for sec in
                                                       metadata_cache['valid_sections'][stack]]
        except:
            pass

        try:
            metadata_cache['valid_sections_all'][stack] = [sec for sec, fn in metadata_cache['sections_to_filenames'][stack].iteritems() if not is_invalid(fn=fn)]
            metadata_cache['valid_filenames_all'][stack] = [fn for sec, fn in metadata_cache['sections_to_filenames'][stack].iteritems() if not is_invalid(fn=fn)]
        except:
            pass

        try:
            metadata_cache['image_shape'][stack] = DataManager.get_image_dimension(stack)
        except:
            pass


generate_metadata_cache()

def resolve_actual_setting(setting, stack, fn=None, sec=None):
    """Take a possibly composite setting index, and return the actual setting index according to fn."""

    if stack in all_nissl_stacks:
        stain = 'nissl'
    elif stack in all_ntb_stacks:
        stain = 'ntb'
    elif stack in all_alt_nissl_ntb_stacks:
        if fn is None:
            assert sec is not None
            fn = metadata_cache['sections_to_filenames'][stack][sec]
        if fn.split('-')[1][0] == 'F':
            stain = 'ntb'
        elif fn.split('-')[1][0] == 'N':
            stain = 'nissl'
        else:
            raise Exception('Must be either ntb or nissl.')

    if setting == 12:
        setting_nissl = 2
        setting_ntb = 10

    if setting == 12:
        if stain == 'nissl':
            setting_ = setting_nissl
        else:
            setting_ = setting_ntb
    else:
        setting_ = setting

    return setting_
