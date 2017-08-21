"""
This module stores static meta information.
"""
import os

########### Data Directories #############

import subprocess
hostname = subprocess.check_output("hostname", shell=True).strip()


if hostname == 'yuncong-MacbookPro':
    print 'Setting environment for Local Macbook Pro'
    HOST_ID = 'localhost'

    # REPO_DIR = '/home/yuncong/Brain' # use os.environ['REPO_DIR'] instead
    REPO_DIR = os.environ['REPO_DIR']
    ROOT_DIR = '/home/yuncong'
    DATA_ROOTDIR = '/media/yuncong/YuncongPublic/'
    THUMBNAIL_DATA_ROOTDIR = ROOT_DIR

    RAW_DATA_DIR = os.path.join(ROOT_DIR, 'CSHL_data')
    DATA_DIR = os.path.join(DATA_ROOTDIR, 'CSHL_data_processed')
    THUMBNAIL_DATA_DIR = os.path.join(THUMBNAIL_DATA_ROOTDIR, 'CSHL_data_processed')

    VOLUME_ROOTDIR = '/home/yuncong/CSHL_volumes'
    MESH_ROOTDIR =  '/home/yuncong/CSHL_meshes'
    REGISTRATION_PARAMETERS_ROOTDIR = '/home/yuncong/CSHL_registration_parameters'
    # ANNOTATION_ROOTDIR = '/home/yuncong/CSHL_data_labelings_losslessAlignCropped'
    ANNOTATION_ROOTDIR = '/home/yuncong/CSHL_labelings_v3'

    S3_DATA_BUCKET = 'mousebrainatlas-data'
    S3_DATA_DIR = 'CSHL_data_processed'
    S3_RAWDATA_BUCKET = 'mousebrainatlas-rawdata'

    CLASSIFIER_SETTINGS_CSV = os.path.join(REPO_DIR, 'learning', 'classifier_settings.csv')
    DATASET_SETTINGS_CSV = os.path.join(REPO_DIR, 'learning', 'dataset_settings.csv')
    REGISTRATION_SETTINGS_CSV = os.path.join(REPO_DIR, 'registration', 'registration_settings.csv')
    PREPROCESS_SETTINGS_CSV = os.path.join(REPO_DIR, 'preprocess', 'preprocess_settings.csv')
    DETECTOR_SETTINGS_CSV = os.path.join(REPO_DIR, 'learning', 'detector_settings.csv')

    LABELED_NEURONS_ROOTDIR = os.path.join(ROOT_DIR, 'CSHL_labeled_neurons')

elif hostname == 'yuncong-Precision-WorkStation-T7500':
    print 'Setting environment for Precision WorkStation'
    HOST_ID = 'workstation'
    ROOT_DIR = '/home/yuncong/'
    DATA_ROOTDIR = '/media/yuncong/BstemAtlasData'
    # THUMBNAIL_DATA_ROOTDIR = ROOT_DIR
    THUMBNAIL_DATA_ROOTDIR = DATA_ROOTDIR
    RAW_DATA_DIR = DATA_ROOTDIR

    ON_AWS = False
    S3_DATA_BUCKET = 'mousebrainatlas-data'
    S3_RAWDATA_BUCKET = 'mousebrainatlas-rawdata'

    REPO_DIR = os.environ['REPO_DIR']

    DATA_DIR = os.path.join(DATA_ROOTDIR, 'CSHL_data_processed')
    THUMBNAIL_DATA_DIR = os.path.join(THUMBNAIL_DATA_ROOTDIR, 'CSHL_data_processed')
    VOLUME_ROOTDIR = os.path.join(ROOT_DIR, 'CSHL_volumes')
    MESH_ROOTDIR =  '/home/yuncong/CSHL_meshes'

    # annotation_rootdir =  os.path.join(ROOT_DIR, 'CSHL_data_labelings_losslessAlignCropped')
#     annotation_midbrainIncluded_v2_rootdir = '/home/yuncong/CSHL_labelings_v3/'
    PATCH_FEATURES_ROOTDIR = os.path.join(DATA_ROOTDIR, 'CSHL_patch_features')
    PATCH_LOCATIONS_ROOTDIR = os.path.join(DATA_ROOTDIR, 'CSHL_patch_locations')

    SCOREMAP_ROOTDIR = os.path.join(ROOT_DIR, 'CSHL_scoremaps')
    SCOREMAP_VIZ_ROOTDIR = os.path.join(ROOT_DIR, 'CSHL_scoremap_viz')
    SPARSE_SCORES_ROOTDIR = os.path.join(ROOT_DIR, 'CSHL_patch_scores')

    ANNOTATION_ROOTDIR =  os.path.join(ROOT_DIR, 'CSHL_labelings_v3')
    CLF_ROOTDIR =  os.path.join(ROOT_DIR, 'CSHL_classifiers')

    REGISTRATION_PARAMETERS_ROOTDIR = '/home/yuncong/CSHL_registration_parameters'

    CLASSIFIER_SETTINGS_CSV = os.path.join(REPO_DIR, 'learning', 'classifier_settings.csv')
    DATASET_SETTINGS_CSV = os.path.join(REPO_DIR, 'learning', 'dataset_settings.csv')
    REGISTRATION_SETTINGS_CSV = os.path.join(REPO_DIR, 'registration', 'registration_settings.csv')
    PREPROCESS_SETTINGS_CSV = os.path.join(REPO_DIR, 'preprocess', 'preprocess_settings.csv')
    DETECTOR_SETTINGS_CSV = os.path.join(REPO_DIR, 'learning', 'detector_settings.csv')

    MXNET_MODEL_ROOTDIR = os.path.join(ROOT_DIR, 'mxnet_models')

elif hostname.startswith('ip'):
    print 'Setting environment for AWS compute node'
    HOST_ID = 'ec2'

    if 'ROOT_DIR' in os.environ:
        ROOT_DIR = os.environ['ROOT_DIR']
    else:
        ROOT_DIR = '/shared'

    if 'DATA_ROOTDIR' in os.environ:
        DATA_ROOTDIR = os.environ['DATA_ROOTDIR']
    else:
        DATA_ROOTDIR = '/shared'
        
    if 'THUMBNAIL_DATA_ROOTDIR' in os.environ:
        THUMBNAIL_DATA_ROOTDIR = os.environ['THUMBNAIL_DATA_ROOTDIR']
    else:
        THUMBNAIL_DATA_ROOTDIR = '/shared'
    
    ON_AWS = True
    S3_DATA_BUCKET = 'mousebrainatlas-data'
    S3_DATA_BUCKET_Xiang = 'mousebrainatlas-xiang'
    S3_RAWDATA_BUCKET = 'mousebrainatlas-rawdata'
    S3_DATA_DIR = 'CSHL_data_processed'
    REPO_DIR = os.environ['REPO_DIR']
    RAW_DATA_DIR = os.path.join(ROOT_DIR, 'CSHL_data')
    DATA_DIR = os.path.join(DATA_ROOTDIR, 'CSHL_data_processed')
    THUMBNAIL_DATA_DIR = os.path.join(THUMBNAIL_DATA_ROOTDIR, 'CSHL_data_processed')
    VOLUME_ROOTDIR = os.path.join(ROOT_DIR, 'CSHL_volumes')
    # SCOREMAP_VIZ_ROOTDIR = '/shared/CSHL_scoremap_viz_Sat16ClassFinetuned_v2'
    ANNOTATION_ROOTDIR = os.path.join(ROOT_DIR, 'CSHL_labelings_v3')
    ANNOTATION_VIZ_ROOTDIR = os.path.join(ROOT_DIR, 'CSHL_annotation_viz')
    # SVM_ROOTDIR = '/shared/CSHL_patch_features_Sat16ClassFinetuned_v2_classifiers/'
    # SVM_NTBLUE_ROOTDIR = '/shared/CSHL_patch_features_Sat16ClassFinetuned_v2_classifiers_neurotraceBlue/'
    PATCH_FEATURES_ROOTDIR = os.path.join(DATA_ROOTDIR, 'CSHL_patch_features')
    PATCH_LOCATIONS_ROOTDIR = os.path.join(DATA_ROOTDIR, 'CSHL_patch_locations')
    SCOREMAP_ROOTDIR = os.path.join(ROOT_DIR, 'CSHL_scoremaps')
    SCOREMAP_VIZ_ROOTDIR = os.path.join(ROOT_DIR, 'CSHL_scoremap_viz')
    SPARSE_SCORES_ROOTDIR = os.path.join(ROOT_DIR, 'CSHL_patch_scores')
    REGISTRATION_PARAMETERS_ROOTDIR = os.path.join(ROOT_DIR, 'CSHL_registration_parameters')
    REGISTRATION_VIZ_ROOTDIR = os.path.join(ROOT_DIR, 'CSHL_registration_visualization')
    # SPARSE_SCORES_ROOTDIR = '/shared/CSHL_patch_Sat16ClassFinetuned_v2_predictions'
    # SCOREMAPS_ROOTDIR = '/shared/CSHL_lossless_scoremaps_Sat16ClassFinetuned_v2'
    # HESSIAN_ROOTDIR = '/shared/CSHL_hessians/'
    ELASTIX_BIN = 'elastix'
    KDU_EXPAND_BIN = '/home/ubuntu/KDU79_Demo_Apps_for_Linux-x86-64_170108/kdu_expand'
    CELLPROFILER_EXEC = 'python /shared/CellProfiler/CellProfiler.py' # /usr/local/bin/cellprofiler
    CELLPROFILER_PIPELINE_FP = '/shared/CSHL_cells_v2/SegmentCells.cppipe'

    if 'CELLS_ROOTDIR' in os.environ:
        CELLS_ROOTDIR = os.environ['CELLS_ROOTDIR']
    else:
        CELLS_ROOTDIR = os.path.join(ROOT_DIR, 'CSHL_cells_v2')
    
    TYPICAL_CELLS_ROOTDIR = os.path.join(ROOT_DIR, 'blob_matching_atlas', 'typical_cells')
    DETECTED_CELLS_ROOTDIR = os.path.join(CELLS_ROOTDIR, 'detected_cells')
    CELL_EMBEDDING_ROOTDIR = os.path.join(CELLS_ROOTDIR, 'embedding')
    D3JS_ROOTDIR = os.path.join(CELLS_ROOTDIR, 'd3js')
    CELL_FEATURES_CLF_ROOTDIR = os.path.join(CELLS_ROOTDIR, 'classifiers')

    CLF_ROOTDIR = os.path.join(ROOT_DIR, 'CSHL_classifiers')

    CLASSIFIER_SETTINGS_CSV = os.path.join(REPO_DIR, 'learning', 'classifier_settings.csv')
    DATASET_SETTINGS_CSV = os.path.join(REPO_DIR, 'learning', 'dataset_settings.csv')
    REGISTRATION_SETTINGS_CSV = os.path.join(REPO_DIR, 'registration', 'registration_settings.csv')
    PREPROCESS_SETTINGS_CSV = os.path.join(REPO_DIR, 'preprocess', 'preprocess_settings.csv')
    DETECTOR_SETTINGS_CSV = os.path.join(REPO_DIR, 'learning', 'detector_settings.csv')

    MXNET_MODEL_ROOTDIR = os.path.join(ROOT_DIR, 'mxnet_models')

    LABELED_NEURONS_ROOTDIR = os.path.join(ROOT_DIR, 'CSHL_labeled_neurons')
    
    CSHL_SPM_ROOTDIR = os.path.join(ROOT_DIR, 'CSHL_SPM')

else:
    print 'Setting environment for Brainstem workstation'

#################### Name conversions ##################

def parse_label(label):
    """
    Args:
        a class label

    Returns:
        (structure name, side, surround margin, surround structure name)
    """
    import re
    try:
        m = re.match("([0-9a-zA-Z]*)(_(L|R))?(_surround_([0-9]+))?(_([0-9a-zA-Z]*))?", label)
    except:
        raise Exception("Parse label error: %s" % label)
    g = m.groups()
    structure_name = g[0]
    side = g[2]
    surround_margin = g[4]
    surround_structure_name = g[6]
    return structure_name, side, surround_margin, surround_structure_name

is_sided_label = lambda label: parse_label(label)[1] is not None
is_surround_label = lambda label: parse_label(label)[2] is not None
get_side_from_label = lambda label: parse_label(label)[1]
get_margin_from_label = lambda label: parse_label(label)[2]

def compose_label(structure_name, side=None, surround_margin=None, surround_structure_name=None):
    label = structure_name
    if side is not None:
        label += '_' + side
    if surround_margin is not None:
        label += '_surround_' + surround_margin
    if surround_structure_name is not None:
        label += '_' + surround_structure_name
    return label

def convert_to_unsided_label(label):
    structure_name, side, surround_margin, surround_structure_name = parse_label(label)
    return compose_label(structure_name, side=None, surround_margin=surround_margin, surround_structure_name=surround_structure_name)

def convert_to_nonsurround_label(name):
    return convert_to_nonsurround_name(name)

    # return convert_name_to_unsided(name)

# def convert_name_to_unsided(name):
#     if '_' not in name:
#         return name
#     else:
#         return convert_to_original_name(name)

def convert_to_left_name(name):
    return convert_to_unsided_label(name) + '_L'

def convert_to_right_name(name):
    return convert_to_unsided_label(name) + '_R'

def convert_to_original_name(name):
    return name.split('_')[0]

def convert_to_nonsurround_name(name):
    if is_surround_label(name):
        import re
        m = re.match('(.*?)_surround_.*', name)
        return m.groups()[0]
    else:
        return name

def convert_to_surround_name(name, margin=None, suffix=None):

    elements = name.split('_')
    if margin is None:
        if len(elements) > 1 and elements[1] == 'surround':
            if suffix is not None:
                return elements[0] + '_surround_' + suffix
            else:
                return elements[0] + '_surround'
        else:
            if suffix is not None:
                return name + '_surround_' + suffix
            else:
                return name + '_surround'
    else:
        if len(elements) > 1 and elements[1] == 'surround':
            if suffix is not None:
                return elements[0] + '_surround_' + str(margin) + '_' + suffix
            else:
                return elements[0] + '_surround_' + str(margin)
        else:
            if suffix is not None:
                return name + '_surround_' + str(margin) + '_' + suffix
            else:
                return name + '_surround_' + str(margin)


#######################################

from pandas import read_csv
dataset_settings = read_csv(DATASET_SETTINGS_CSV, header=0, index_col=0)
classifier_settings = read_csv(CLASSIFIER_SETTINGS_CSV, header=0, index_col=0)
registration_settings = read_csv(REGISTRATION_SETTINGS_CSV, header=0, index_col=0)
preprocess_settings = read_csv(PREPROCESS_SETTINGS_CSV, header=0, index_col=0)
detector_settings = read_csv(DETECTOR_SETTINGS_CSV, header=0, index_col=0)
windowing_settings = {1: {"patch_size": 224, "spacing": 56}, 
                      2: {'patch_size':224, 'spacing':56, 'comment':'larger margin'},
                     3: {'patch_size':224, 'spacing':32, 'comment':'smaller spacing'},
                     4: {'patch_size':224, 'spacing':128, 'comment':'smaller spacing'},
                     5: {'patch_size':224, 'spacing':64, 'comment':'smaller spacing'}}

############ Class Labels #############

paired_structures = ['5N', '6N', '7N', '7n', 'Amb', 'LC', 'LRt', 'Pn', 'Tz', 'VLL', 'RMC', 'SNC', 'SNR', '3N', '4N',
                    'Sp5I', 'Sp5O', 'Sp5C', 'PBG', '10N', 'VCA', 'VCP', 'DC']
singular_structures = ['AP', '12N', 'RtTg', 'sp5', 'outerContour', 'SC', 'IC']
all_known_structures = paired_structures + singular_structures
all_known_structures_sided = sum([[n] if n in singular_structures
                        else [convert_to_left_name(n), convert_to_right_name(n)]
                        for n in all_known_structures], [])
#all_known_structures_sided_surround_only = [convert_to_surround_name(s, margin='x1.5') for s in all_known_structures_sided]
all_known_structures_sided_surround_only = [convert_to_surround_name(s, margin=200) for s in all_known_structures_sided]
all_known_structures_sided_with_surround = sorted(all_known_structures_sided + all_known_structures_sided_surround_only)
all_structures_with_classifiers = sorted([l for l in all_known_structures if l not in {'outerContour', 'sp5'}])


linear_landmark_names_unsided = ['outerContour']
volumetric_landmark_names_unsided = list(set(paired_structures + singular_structures) - set(linear_landmark_names_unsided))
all_landmark_names_unsided = volumetric_landmark_names_unsided + linear_landmark_names_unsided

labels_unsided = volumetric_landmark_names_unsided + linear_landmark_names_unsided
labels_unsided_indices = dict((j, i+1) for i, j in enumerate(labels_unsided))  # BackG always 0

labelMap_unsidedToSided = dict([(name, [name+'_L', name+'_R']) for name in paired_structures] + \
                            [(name, [name]) for name in singular_structures])

labelMap_sidedToUnsided = {n: nu for nu, ns in labelMap_unsidedToSided.iteritems() for n in ns}

from itertools import chain
labels_sided = list(chain(*(labelMap_unsidedToSided[name_u] for name_u in labels_unsided)))
labels_sided_indices = dict((j, i+1) for i, j in enumerate(labels_sided)) # BackG always 0

############ Physical Dimension #############

# section_thickness = 20 # in um
SECTION_THICKNESS = 20 # in um
# xy_pixel_distance_lossless = 0.46
XY_PIXEL_DISTANCE_LOSSLESS = 0.46
XY_PIXEL_DISTANCE_TB = XY_PIXEL_DISTANCE_LOSSLESS * 32 # in um, thumbnail

#######################################

all_nissl_stacks = ['MD585', 'MD589', 'MD590', 'MD591', 'MD592', 'MD593', 'MD594', 'MD595', 'MD598', 'MD599', 'MD602', 'MD603']
all_ntb_stacks = ['MD635']
all_alt_nissl_ntb_stacks = ['MD653', 'MD652', 'MD642']
all_alt_nissl_tracing_stacks = ['MD657', 'MD658', 'MD661', 'MD662']
# all_stacks = all_nissl_stacks + all_ntb_stacks
all_stacks = all_nissl_stacks + all_ntb_stacks + all_alt_nissl_ntb_stacks + all_alt_nissl_tracing_stacks
all_annotated_nissl_stacks = ['MD585', 'MD589', 'MD594']
all_annotated_ntb_stacks = ['MD635']
all_annotated_stacks = all_annotated_nissl_stacks + all_annotated_ntb_stacks

#######################################

import multiprocessing
NUM_CORES = multiprocessing.cpu_count()

############## Colors ##############

from utilities2015 import high_contrast_colors
hc_perm = [ 0,  5, 28, 26, 12, 11,  4,  8, 25, 22,  3,  1, 20, 19, 27, 13, 24,
       17, 16, 15,  7, 14, 21, 18, 23,  2, 10,  9,  6]
high_contrast_colors = [high_contrast_colors[i] for i in hc_perm]
name_sided_to_color = {s: high_contrast_colors[i%len(high_contrast_colors)] 
                     for i, s in enumerate(all_known_structures_sided) }
name_unsided_to_color = {s: high_contrast_colors[i%len(high_contrast_colors)] 
                     for i, s in enumerate(all_known_structures) }
stack_to_color = {n: high_contrast_colors[i%len(high_contrast_colors)] for i, n in enumerate(all_stacks)}
