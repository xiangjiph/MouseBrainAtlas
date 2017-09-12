import sys, os, time, datetime
from skimage.filters import threshold_otsu
import skimage
from scipy import ndimage as ndi
from scipy.signal import argrelmax
from scipy.stats import linregress
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from shapely.geometry import Polygon as Polygon
from cell_utilities import *
from data_manager import *
from utilities2015 import *
from metadata import *

### Constants ###
PI = 3.1415926535897932384626
REGIONPROPS_BUILTIN = ['centroid','orientation', 'eccentricity','area','moments_hu','bbox','equivalent_diameter','label','local_centroid','major_axis_length','solidity','minor_axis_length','perimeter']
REGIONPROPS_Euclid_dis = ['centroid','eccentricity','area', 'equivalent_diameter', 'major_axis_length','solidity','minor_axis_length', 'perimeter', 'compactness', 'euclid']
### Setting parameters###
#scan_parameters = {}
#scan_parameters['patch_size'] = 448
#scan_parameters['patch_half_size'] = scan_parameters['patch_size']/2
#scan_parameters['stride'] = 112
#scan_parameters['o_clear_border'] = True
#scan_parameters['o_relabel'] = True
#scan_parameters['o_fix_scan_size'] = True
#scan_parameters['scan_section_range'] = 1
#scan_parameters['scan_size'] = 112
#scan_parameters['scan_size_coeff'] = 5
#scan_parameters['builtInProps'] = ['centroid','orientation', 'eccentricity','area','orientation','moments_hu','bbox','equivalent_diameter','label','local_centroid','major_axis_length','solidity','minor_axis_length','perimeter','solidity']
#scan_parameters['prop_to_save'] = ['coords','moments_hu','centroid','area','eccentricity','equivalent_diameter']

### File management ###
# save_folder_path = '/shared/MouseBrainAtlasXiang/XJ/Output/detect_cell_alternatives_output/';
# sys.stderr.write('Path: '+save_folder_path)
def fun_create_folder(save_folder_path=None):
    if save_folder_path == None:
        save_folder_path = '/shared/MouseBrainAtlasXiang/XJ/Output/'+ datetime.datetime.now().strftime('%H_%M_%S_%B_%d_%Y')
    if os.path.isdir(save_folder_path):
        sys.stderr.write('Folder already exists: ' + save_folder_path)
    else:
        os.makedirs(save_folder_path)
        sys.stderr.write('Folder created: ' + save_folder_path)

def fun_construct_filename(define_list):
    """
    get_finelname(list_of_definition_component): return a string concaining filename definition, seperated by '_' 
    """
    filename_head = ''
    filename_element = map(str, define_list)
    num_element = len(filename_element)
    for idx in range(num_element-1):
        filename_head = filename_head + filename_element[idx] + '_'
    filename_head = filename_head + filename_element[num_element-1]
    return filename_head


### Visualization ###

def fun_viz_classes(data_dic,patch_loc_list,nornmalized=True,transposed=True,patch_stride=56):
    """
    vis_classes(data_dic,patch_loc_index,nornmalized=True,transposed=True,patch_stride=56)
    data_dic = {key_value : indexes}
    patch_loc_index:
    require: numpy, sys
    """
    # Get an array of the index location of each patches
    patch_center_pixel_index = patch_loc_list/patch_stride - 2
    patch_loc_plot = np.zeros(np.max(patch_center_pixel_index,axis=0)+1)
    for tempKey in data_dic.viewkeys():
        for temoLoc in patch_center_pixel_index[data_dic[tempKey]]:
            try:
                patch_loc_plot[temoLoc[0], temoLoc[1]]=tempKey
            except:
                sys.stderr.write('Index location out of range'+str(temoLoc));
    if nornmalized:
        try:
            patch_loc_plot = fun_rescale_grayscale_image(patch_loc_plot)
        except:
            #sys.stderr('Cannot rescale the grayscale image. Check if function rescale_grayscale_image is available.')
            print('Cannot rescale the grayscale image. Check if function rescale_grayscale_image is available.')
    if transposed:
        patch_loc_plot = np.transpose(patch_loc_plot)
    return patch_loc_plot

### Compute patch features ###

def fun_patch_features_DT(patch_image_BW, patch_area=50176,showImQ=False):
    """
    patch_feature_DT(patch_image_BW, showImQ=False): Input binary patch images, output features extracted
    from the distance transformation of the pathc image, including MaxDis, LFslope, LFicpt, LFpV and LFstd. 
    
    """
    distance_im = ndi.morphology.distance_transform_edt(patch_image_BW)
    # Generate bins for histogram
    n_dis_hist_bins = 16; 
    dis_hist_max = np.max(distance_im.flatten())
    dis_hist_min = np.min(distance_im.flatten())
    dis_hist_bins = np.linspace(dis_hist_min, dis_hist_max, num=n_dis_hist_bins+1)
    distance_im_his = np.histogram(distance_im.flatten(),bins=dis_hist_bins)
    # Get x coordinate defined as the center of each bin by moving average
    dis_hist_bins_mid = np.convolve(distance_im_his[1],np.ones((2,))/2,mode='valid')
    
    dis_hist_1st_bin_NumRatio = distance_im_his[0][0]/float(np.sum(distance_im_his[0]))
    dis_hist_last_bin_AreaRatio = distance_im_his[0][-1] * dis_hist_bins_mid[-1] / float(patch_area)
    # Transfrom for linear regression, start from the second element. (*First element is very large)
    dis_hist_count_log = np.log10(distance_im_his[0][1:])
    dis_hist_LF = linregress(dis_hist_bins_mid[1:], dis_hist_count_log)
    features_DT = {'MaxDis':dis_hist_max,'LFslope':dis_hist_LF.slope,'LFicpt':dis_hist_LF.intercept,
                   'LFpV':dis_hist_LF.pvalue,'LFstd':dis_hist_LF.stderr}

    if showImQ:
        fig_dismap = plt.figure()
        # fig_dismap = plt.imshow(distance_im, cmap=plt.cm.gray)
        fig_dismap = plt.imshow(distance_im)
        dis_dismap = plt.colorbar()
        fig_stat = plt.figure()
        fig_stat = plt.scatter(dis_hist_bins_mid,np.log10(distance_im_his[0]));
        fig_stat = plt.hold
        fig_stat = plt.plot(dis_hist_bins_mid[1:],dis_hist_LF.slope * dis_hist_bins_mid[1:] + dis_hist_LF.intercept,color='red');
        fig_stat = plt.title('slope = '+str(dis_hist_LF.slope)+'  Max_dist = '+str(dis_hist_max) + '\n  1stBinRatio = ' + str(dis_hist_1st_bin_NumRatio)+ '  LastBinAreaRatio = ' + str(dis_hist_last_bin_AreaRatio))
        fig_stat = plt.xlabel('Distance')
        fig_stat = plt.ylabel('Log10(#Pixel)')
    return features_DT, distance_im

### Image Processing ### 
def fun_rescale_grayscale_image(inputdata,invcol=False,pvmax=255):
    inputdata = np.array(inputdata);
    vmin = np.percentile(inputdata, 0);
    vmax = np.percentile(inputdata, 100);
    rescale_image = (inputdata - vmin ) / (vmax - vmin);
    rescale_image = np.maximum(np.minimum(rescale_image, 1), 0)
    if pvmax == 255:
        rescale_image = skimage.img_as_ubyte(rescale_image)
    elif pvmax ==1:
        pass
    if invcol:
        rescale_image = 255 - rescale_image
    return rescale_image

# Function to find the threshold by gradient alignemnt
def fun_threshold_gradAlig(image,scanrange=(160,240),step=5,o_size_gaussfilt=1,o_return_scores=False,show_score_plot=True,method='avg_dot_pdt'):
    """
        image: a grayscale image(0-255)
        step: step for grayscale scan. can be integer number larger than 0
        o_size_gaussfilt: size of the gaussian filter. 
                          Not apply the gaussian filter to the image if equals 0
        method: (1)tot_dot_pdt; (2)avg_cos
        package needed: scipy.ndimage as ndi; numpy as np; matplotlib.pyplot as plt;  
        
    """
    if o_size_gaussfilt > 0:
        tempImage = (ndi.filters.gaussian_filter(image.astype(np.float), o_size_gaussfilt)) # ndi retuens the same type of array as image(uint array)
    else:
        tempImage = image.astype(np.float)
    gradX, gradY = np.gradient(tempImage)
    image_size = image.size
    temp_TH_aggrement = {}
    
    for tempTH in np.arange(scanrange[0],scanrange[1],step):
        tempGx, tempGy = np.gradient((tempImage > tempTH).astype(np.float))
        nz = np.logical_or(tempGx != 0, tempGy != 0)
        if np.count_nonzero(nz) == 0:
            temp_TH_aggrement[tempTH] = 0
        else:
            if method == 'tot_dot_pdt':
                    temp_TH_aggrement[tempTH] = np.sum(tempGx[nz]*gradX[nz] + tempGy[nz]*gradY[nz])
            if method == 'avg_dot_pdt':
                temp_TH_aggrement[tempTH] = np.average(tempGx[nz]*gradX[nz] + tempGy[nz]*gradY[nz])
            if method == 'avg_cos':
                grad_norm = np.sqrt( gradX[nz] ** 2 + gradY[nz] ** 2 )
                tempG_norm = np.sqrt( tempGx[nz] ** 2 + tempGy[nz] ** 2 )
                temp_TH_aggrement[tempTH] = np.average((tempGx[nz]*gradX[nz] + tempGy[nz]*gradY[nz])/(tempG_norm * grad_norm + 0.000000000000001))
    tempKeys, tempValues = zip(*sorted(temp_TH_aggrement.items()))
    grad_threshold = tempKeys[np.argmax(tempValues)]
    
    if show_score_plot==True:
        fig_grad_THscan_score = plt.figure()
        fig_grad_THscan_score = plt.plot(tempKeys, tempValues)
        fig_grad_THscan_score = plt.xlabel('Grayscale threshold')
        fig_grad_THscan_score = plt.ylabel('Score')
        fig_grad_THscan_score = plt.title('Score method used: '+method+'\nBest threshold = %s' % grad_threshold)
        fig_grad_THscan_score = plt.grid(True)
    if o_return_scores==True:
        return grad_threshold,[tempKeys,tempValues]
    else:
        return grad_threshold    

    
def fun_radius_bbox(min_0, min_1, max_0, max_1):
    """
    radisu_bbox(min_0, min_1, max_0, max_1), correspond to the order of regionprops.bbox tuple
    """
    radius = 0.5 * ((max_1 - min_1)**2 + (max_0 - min_0)**2) ** 0.5
    return radius

def fun_crop_images(image, min_0, min_1, max_0, max_1, margin=0,im0max=10000,im1max=10000):
    min_0 = max(min_0-margin,0)
    max_0 = min(max_0+margin,im0max)
    min_1 = max(min_1-margin,0)
    max_1 = min(max_1+margin,im1max)
    crop_image = image[min_0:max_0, min_1:max_1].copy()
#     print((min_0, min_1, max_0, max_1))
    return crop_image

def fun_mxmx_to_mmxx(min_0,max_0,min_1,max_1):
    return (min_0, min_1, max_0, max_1)

def fun_mmxx_to_mxmx(min_0, min_1, max_0, max_1):
    return (min_0, max_0, min_1, max_1)

def fun_scan_range(cloc,radius,im1max=10000,im0max=10000,o_form='1D'):
    cloc = np.array(cloc);
    min_0 = int(max(np.round(cloc - radius)[0],0))
    min_1 = int(max(np.round(cloc - radius)[1],0))
    max_0 = int(min(np.round(cloc + radius)[0],im0max))
    max_1 = int(min(np.round(cloc + radius)[1],im1max))
    local_cloc = (int(cloc[0] - min_0), int(cloc[1] - min_1))
    if o_form == '1D':
        return (min_0, min_1, max_0, max_1), local_cloc
    elif o_form == '2D':
        return np.array([[min_0,min_1],[min_0,max_1],[max_0,max_1],[max_0,min_1]]), local_cloc

def fun_local_distance(blob_loc_tuple, local_cloc_tuple):
    r = ((blob_loc_tuple[0] - local_cloc_tuple[0]) ** 2 + ((blob_loc_tuple[1] - local_cloc_tuple[1]) ** 2) ) ** 0.5
    return r

def fun_similarity(oriIprops,nextIprops,distance_type='euclid',return_type='list'):
    """
    Args: 
        oriIprops: region property of single blob, can be float, int or numpy.ndarray
        nextIprops: region property of single blob or a list/numpy.ndarray of region properties. Components should be the same type as oriIprops. 
        distance_type: specify the way to measure the similarity, can be region property like 'area', 'moment_hu'. 
    Returns:
        similarity between oriIprops and nextIprops. A (list of) float(s) between [0,1], depends on the input nextIprops.
    """
    similarity = [];
    if type(nextIprops) not in [list, np.ndarray]:
        nextIprops = [nextIprops]
    if distance_type=='moments_hu':
        if np.shape(nextIprops) == (7,):
            nextIprops = [nextIprops]
    num_blob = len(nextIprops);
    pi = 3.1415926
    for i in range(num_blob):
        if distance_type in REGIONPROPS_Euclid_dis:
            difference = abs(oriIprops - nextIprops[i])/(abs(float(max(oriIprops, nextIprops[i]))) + 0.000000000001)
        elif distance_type == 'moments_hu':
            difference = np.abs(np.abs(oriIprops) - np.abs(np.array(nextIprops[i],dtype=np.float)))/np.abs( np.max(np.abs(np.vstack((oriIprops,nextIprops[i]))),axis=0) + 10**(-16)) 
        elif distance_type == 'orientation':
            diff_angle = abs(oriIprops - nextIprops[i])
            diff_angle = min(diff_angle, pi - diff_angle )
            similarity.append(np.cos(diff_angle))
            continue
        elif distance_type == 'angular':
            diff_angle = abs(oriIprops - nextIprops[i])
            diff_angle = min(diff_angle, pi - diff_angle )
            difference = np.cos(diff_angle) 
        similarity.append( 1 - difference)
    if return_type == 'list':
        return similarity
    elif num_blob == 1:
        similarity = similarity[0]
        return similarity

def fun_reconstruct_labeled_image(cell_global_coord,oriImL0, oriImL1, crop_range=None, op_clear_border=True,op_relabel=True):
    """ cell_global_coord = list of coordinate of the global index position of all the pixel in each blob
        oriImL1, oriImL0 = metadata_cache['image_shape][stack]
        crop_range = (min0, max0, min1, max1)
        return: labeled_image, blob_prop_List
    """
    cell_numbers = len(cell_global_coord);
    tempLabeledImage = np.zeros([oriImL0,oriImL1],dtype=np.int32)
    
    for tempBlobIndex in range(cell_numbers):
        tempBlobCoor = cell_global_coord[tempBlobIndex]
        tempLabeledImage[tempBlobCoor[:,0],tempBlobCoor[:,1]] = tempBlobIndex + 1
    if crop_range is not None:
        crop_0_min = crop_range[0]
        crop_0_max = crop_range[1]
        crop_1_min = crop_range[2]
        crop_1_max = crop_range[3]
        tempLabeledImage = tempLabeledImage[crop_0_min:crop_0_max, crop_1_min:crop_1_max]
    if op_clear_border:
        tempLabeledImage = skimage.segmentation.clear_border(tempLabeledImage)
        
    if op_relabel:
        im_label_ori = tempLabeledImage
        tempLabeledImage = skimage.measure.label(tempLabeledImage > 0)
        im_blob_prop = skimage.measure.regionprops(tempLabeledImage)
        return tempLabeledImage, im_blob_prop, im_label_ori
    else:
        im_blob_prop = skimage.measure.regionprops(tempLabeledImage)
        return tempLabeledImage, im_blob_prop

def fun_blobs_in_polygen(blob_centroid_list, contour_vertice_coor_array, crop_min_list=[0,0], coor_order='rc'):
    """contour_vertice_coor_array = np.array([[point1_row, point1_col]; rc for Yuncong's data, cr for numpy"""
    if coor_order == 'rc':
        contourPath = matplotlib.path.Path(contour_vertice_coor_array[:,[1,0]] - crop_min_list) # Yuncong uses (row, col) while numpy use 
    elif coor_order == 'cr':
        contourPath = matplotlib.path.Path(contour_vertice_coor_array[:,[0,1]] - crop_min_list) # Yuncong uses (row, col) while numpy use 
    return contourPath.contains_points(blob_centroid_list)

def fun_blobs_out_polygen(blob_centroid_list, contour_vertice_coor_array, crop_min_list=[0,0], margin=0, coor_order='rc'):
    """contour_vertice_coor_array = np.array([[point1_row, point1_col]; rc for Yuncong's data, cr for numpy"""
    if coor_order == 'rc':
        contour_polygon = Polygon(contour_vertice_coor_array[:,[1,0]] - crop_min_list)
    elif coor_order == 'cr':
        contour_polygon = Polygon(contour_vertice_coor_array[:,[0,1]] - crop_min_list)
    contour_polygon_with_margin = contour_polygon.buffer(margin, resolution=2)
    contour_polygon_with_margin = matplotlib.path.Path(list(contour_polygon_with_margin.exterior.coords))
    return np.logical_not(contour_polygon_with_margin.contains_points(blob_centroid_list))
       
    
def fun_collect_typical_blobs(im_blob_prop,im_label, section, scan_parameters):
    """ Output: typical_blobs: List of [section, blobID, im_blob_prop[section][blobID]];
    matched_paris: List of [section,blobID,im_blob_prop[section][blobID],tempSec, matched_blob_ID, matched_blob_props,matched_blob_similarity_matrix]    
    """
    typical_blobs = []
    matched_paris = []
    scan_range = scan_parameters['scan_section_range']
    scan_section = range(section - scan_range, section + scan_range + 1)
    scan_section.remove(section)
    im0max = scan_parameters['im0max']
    im1max = scan_parameters['im1max']

    prop = scan_parameters['prop']
    prop_for_comparison = scan_parameters['prop_for_comparison']

    compare_weight = scan_parameters['compare_weight']
    o_simil_threshold = scan_parameters['similarity_threshold']

    o_fix_scan_size = scan_parameters['o_fix_scan_size']
    o_scan_size_coeff = scan_parameters['scan_size_coeff']
    o_scan_size = scan_parameters['scan_size']  
    secList_in_BlobPropDic = im_blob_prop.keys()
    n_blobs = {}
    blobs_idx_dic = {}
    blobs_centroids_dic = {}
    for tempSec in secList_in_BlobPropDic:
        n_blobs[tempSec] = len(im_blob_prop[tempSec])
        blobs_idx_dic[tempSec] = np.arange(0, n_blobs[tempSec], dtype=np.int32)
        blobs_centroids_dic[tempSec] = np.array([im_blob_prop[tempSec][tempBID]['centroid'] for tempBID in blobs_idx_dic[tempSec]])


    if set(scan_section).issubset(set(secList_in_BlobPropDic)):
        pass
    else:
        print('Warrning: Scaned section(s) not included in input im_blob_prop')

    for blobID in range(n_blobs[section]):
        if (blobID % 1000 == 0):
            print('Section %d Finished percentage: %f'%(section, (float(blobID)*100 / n_blobs[section]) ))

        temp_curr_blob_props = im_blob_prop[section][blobID]
        tempB1_idx_loc = temp_curr_blob_props['centroid']
        if o_fix_scan_size:
            temp_next_sec_range, local_cloc = fun_scan_range(tempB1_idx_loc,o_scan_size,im0max=im0max,im1max=im1max,o_form='2D')
            temp_next_sec_range_1D,_ = fun_scan_range(tempB1_idx_loc,o_scan_size,im0max=im0max,im1max=im1max)
        else:
            temp_next_sec_range, local_cloc = fun_scan_range(tempB1_idx_loc,o_scan_size_coeff*fun_radius_bbox(*temp_curr_blob_props.bbox),im0max=im0max,im1max=im1max,o_form='2D')
            temp_next_sec_range_1D,_ = fun_scan_range(tempB1_idx_loc,o_scan_size_coeff*fun_radius_bbox(*temp_curr_blob_props.bbox),im0max=im0max,im1max=im1max)


        for tempSec in scan_section:
            if tempSec not in secList_in_BlobPropDic:
                continue

            # Find blobs at the nearby location in the scaned section
            # Method 1
            tempPath = matplotlib.path.Path(temp_next_sec_range)
            tempBlobInside = tempPath.contains_points(blobs_centroids_dic[tempSec])
            tempBlobInsideIndex = blobs_idx_dic[tempSec][tempBlobInside]            
            # Method 2
#            temp_im = fun_crop_images(im_label[tempSec],*temp_next_sec_range_1D, margin=0,im0max=im0max, im1max=im1max)
#            tempBlobInsideIndex = np.delete(np.unique(temp_im.flatten()),0,axis=0) - 1           
            temp_num_blob = len(tempBlobInsideIndex)
            if temp_num_blob:
                temp_next_sec_blob_prop = np.array(im_blob_prop[tempSec])[tempBlobInsideIndex]
            else:
    #             print('No blobs found in this section')
                continue

            # Get blob properties
            temp_next_blob_props = {}
            for tempProp in prop:
                temp_prop_value = []
                if tempProp=='relative_dict':
                    for blobIndex in range(temp_num_blob):
                        temp_prop_value.append(fun_local_distance(temp_next_sec_blob_prop[blobIndex]['centroid'],local_cloc))
                elif tempProp in ['centroid','eccentricity','area','orientation','moments_hu','bbox','equivalent_diameter']:
                    for blobIndex in range(temp_num_blob):
                        temp_prop_value.append(temp_next_sec_blob_prop[blobIndex][tempProp])
                elif tempProp=='compactness':
                    temp_prop_value.append(temp_next_sec_blob_prop[blobIndex]['perimeter']**2 /
                                           temp_next_sec_blob_prop[blobIndex]['area'] / (4*PI))
                temp_next_blob_props[tempProp] = temp_prop_value
               

            #### Construct similarity matrix ####
            temp_sim = {}
            for temp_prop in prop_for_comparison:
                    temp_sim[temp_prop] = np.array(fun_similarity(temp_curr_blob_props[temp_prop],
                                                             temp_next_blob_props[temp_prop],distance_type=temp_prop))
            temp_sim_matrix = np.column_stack((temp_sim[temp_prop] for temp_prop in prop_for_comparison))

            #### Blob comparison ####
            temp_weighted_sim = np.dot(temp_sim_matrix,compare_weight)
            temp_compare_result = temp_weighted_sim > o_simil_threshold
            if any(temp_compare_result.tolist()):
                typical_blobs.append([section, blobID, im_blob_prop[section][blobID]])
                # list of [section, blobID, blob_properties, scan_section, matched_blob_IDs, matched_blobs_properties, match_blob_similarities]
                #matched_paris.append([section,blobID,im_blob_prop[section][blobID],tempSec, tempBlobInsideIndex[temp_compare_result], im_blob_prop[tempSec][tempBlobInsideIndex[temp_compare_result]],temp_sim_matrix[temp_compare_result,:]])
    return typical_blobs, matched_paris
    # return typical_blobs

def fun_load_data_collect_typical_blobs(sec, scan_parameters,o_save=False):
    scan_section_range = scan_parameters['scan_section_range']
    sec_load_data_list = range(sec - scan_section_range, sec + scan_section_range + 1)
    scan_section = list(sec_load_data_list)
    scan_section.remove(sec)
    secList = scan_parameters['secList']
    stack = scan_parameters['stack']

    cell_centroids = {}
    cell_numbers = {}
    cell_global_coord = {}
    im_blob_prop = {}
    im_label_ori = {}
    im_label = {}
    im_BW = {}
    for tempSec in sec_load_data_list:
        if tempSec in secList:
            cell_global_coord[tempSec] = load_cell_data('coords', stack=stack, sec=tempSec)
            temp_im_label, temp_im_blob_prop, _ = fun_reconstruct_labeled_image(cell_global_coord[tempSec],crop_range= scan_parameters['crop_range_mxmx'], 
                                                                        oriImL0=scan_parameters['oriImL0'],oriImL1=scan_parameters['oriImL1'])
            im_label[tempSec] = temp_im_label
            im_BW[tempSec] = temp_im_label > 0
            im_blob_prop[tempSec] = np.array(temp_im_blob_prop)
        else:
            sys.stderr.write('Warning: missing section %d'%tempSec)
            scan_section.remove(tempSec)
    n_blobs = {tempSec: len(im_blob_prop[tempSec]) for tempSec in im_blob_prop.keys()}
    #print('Scanning section %d'%sec)
    typical_blobs, matched_pairs = fun_collect_typical_blobs(im_blob_prop=im_blob_prop, im_label=im_label, section=sec,scan_parameters=scan_parameters)
    if o_save==True:
        tempFp = get_typical_cell_data_filepath(what='scan_parameters',stack=stack,sec=sec)
        create_if_not_exists(os.path.dirname(tempFp))
        save_pickle(scan_parameters,tempFp)
        regionprop_List = [tempRecord[2] for tempRecord in typical_blobs]
        fun_save_regionprops(regionprop_List=regionprop_List, prop_to_save=scan_parameters['prop_to_save'],stack=stack, sec=sec)
        print('Result saved.')
        return 0
    else:
        return typical_blobs, matched_pairs



def fun_save_regionprops(regionprop_List, prop_to_save, stack, sec, dataType='typical', dataFolderName=None):
    for tempProp in prop_to_save:
        tempProp_data = []
        for record in regionprop_List:
            tempProp_data.append(record[tempProp])
            
        if tempProp == 'coords':
            tempProp_data = map(lambda data: np.array(data, dtype=np.int16), tempProp_data)
        elif tempProp == 'moments_hu':
            tempProp_data = map(lambda data: np.array(data, dtype=np.float32), tempProp_data)
            tempProp_data = np.row_stack(tuple(tempProp_data))
        elif tempProp == 'centroid':
            tempProp_data = map(lambda data: np.array(data, dtype=np.float32), tempProp_data)
            tempProp_data = np.row_stack(tuple(tempProp_data))
        elif tempProp == 'area':
            tempProp_data = np.array(tempProp_data, np.int32)
        elif tempProp == 'eccentricity':
            tempProp_data = np.array(tempProp_data, np.float32)
        elif tempProp == 'equivalent_diameter':
            tempProp_data = np.array(tempProp_data, np.float32)
        elif tempProp == 'compactness':
            tempProp_data = np.array(tempProp_data, np.float32)
        elif tempProp == 'perimeter':
            tempProp_data = np.array(tempProp_data, np.int32)

        tempFp = get_typical_cell_data_filepath(what=tempProp,stack=stack,sec=sec, dataType=dataType, dataFolderName=dataFolderName)
        create_if_not_exists(os.path.dirname(tempFp))
        if tempFp.endswith('.hdf'):
            save_hdf_v2(tempProp_data, fn=tempFp)
        elif tempFp.endswith('.bp'):
            bp.pack_ndarray_file(tempProp_data, tempFp)
        else:
            print 'Unrecognized data type. Save failed.'
    return 0


def fun_vis_typical_blob(stack,sec,o_overlay_on_oriImage=True,o_save_image=True):
    typical_blob_coords = [record[0] for record in load_typical_cell_data(what='coords',stack=stack, sec=sec)]
    scan_parameters = load_typical_cell_data(what='scan_parameters',stack=stack,sec=sec)
    temp_vis_tyblob_in_sec,_,_ = fun_reconstruct_labeled_image(typical_blob_coords,oriImL0=scan_parameters['im0max'],oriImL1=scan_parameters['im1max'])
    
    if o_overlay_on_oriImage==True:
        oriImage = DataManager.get_image_filepath(stack=stack,section=sec,resol='lossless', version='cropped')
        oriImage = imread(oriImage)[scan_parameters['crop_0_min']:scan_parameters['crop_0_max'],scan_parameters['crop_1_min']:scan_parameters['crop_1_max']]
        cell_contour = skimage.measure.find_contours(temp_vis_tyblob_in_sec>0,0)                                                                
        for tempContour in cell_contour:
            cv2.polylines(oriImage, [tempContour[:,::-1].astype(np.int32)], isClosed=True, color=(255,0,0), thickness=3)
        temp_vis_tyblob_in_sec=oriImage
    if o_save_image==True:
        fp = get_typical_cell_data_filepath(what='image',stack=stack,sec=sec)
        imsave(fp,temp_vis_tyblob_in_sec)
        return 'Image saved'
    else:
        return display_image(temp_vis_tyblob_in_sec)
    
    
def fun_polygon_bbox(vertice_list,margin=0):
    vertice_list = np.array(vertice_list)
    min1,min0 = np.min(vertice_list,axis=0) - margin
    max1,max0 = np.max(vertice_list,axis=0) + margin
    return (min0,min1,max0,max1)

def fun_polygons_bbox(bbox_list,margin=0):
    """bbox_list: a list of 4 vertices of bbox """
    bbox_list = np.array(bbox_list)
    min0 = np.min(bbox_list[:,0]) - margin
    min1 = np.min(bbox_list[:,1]) - margin
    max0 = np.max(bbox_list[:,2]) + margin
    max1 = np.max(bbox_list[:,3]) + margin
    return (min0,min1,max0,max1)



def fun_regionprops_compactness(region_prop):
    PI = 3.141592653589793
    return region_prop['perimeter']**2/region_prop['area']/(4*PI)

def fun_regionprops_dic(im_blob_prop,scan_parameters):
    """im_blob_prop: a dict of lists of regionprops;
       scan_parameters: a dict specifying regionprops to get
    """
    import collections
    blob_prop_dic = collections.defaultdict(dict)
    n_blobs = {tempSec: len(im_blob_prop[tempSec]) for tempSec in im_blob_prop.keys()}
    for tempSec in im_blob_prop.keys():
        for tempProp in scan_parameters['prop']:
            tempPropValues = []
            if tempProp in scan_parameters['builtInProps']:
                for tempBlobID in range(n_blobs[tempSec]):
                    tempPropValues.append(im_blob_prop[tempSec][tempBlobID][tempProp])
            elif tempProp == 'compactness':
                for tempBlobID in range(n_blobs[tempSec]):
                    tempPropValues.append(fun_regionprops_compactness(im_blob_prop[tempSec][tempBlobID]))
            blob_prop_dic[tempSec][tempProp] = np.array(tempPropValues)
    return blob_prop_dic


def fun_angle_arc_to_degree(angle):
    PI = 3.141592653589793
    return angle*180.0/PI

def fun_angle_degree_to_arc(angle):
    return float(angle)*PI/180.0

def fun_angle_change_interval(angle,unit='arc'):

    if unit == 'arc':
        PI = 3.141592653589793
        if (angle <= PI/2) and (angle>=0):
            return angle
        elif (- PI/2 <= angle) and (angle <0):
            return angle + PI
    elif unit == 'degree':
        PI = 180
        if (angle>=0) and (angle<=90):
            return angle
        elif (-90<=angle) and (angle<0):
            return angle + PI
    
def fun_get_valid_section_list(stack):
    valid_section_list = []
    for sec in range(*metadata_cache['section_limits'][stack]):
        if not is_invalid(sec=sec, stack=stack):
            valid_section_list.append(sec)
    return valid_section_list
        