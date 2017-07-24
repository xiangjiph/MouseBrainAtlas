import sys, os, time, datetime
from skimage.filters import threshold_otsu
import skimage
from scipy import ndimage as ndi
from scipy.signal import argrelmax
from scipy.stats import linregress
import numpy as np
import matplotlib.pyplot as plt

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
    for filename_element in map(str,define_list):
        filename_head = filename_head + filename_element + '_'
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

def fun_similarity(oriIprops,nextIprops,distance_type='euclid'):
    similarity = [];
    num_blob = len(nextIprops);
    pi = 3.1415926
    for i in range(num_blob):
        if distance_type == 'euclid':
            difference = abs(oriIprops - nextIprops[i])/abs(float(max(oriIprops, nextIprops[i])) + 0.000000000001)
        elif distance_type == 'area':
            difference = abs(oriIprops - nextIprops[i])/(float(max(oriIprops, nextIprops[i])) + 0.000000000001)
        elif distance_type == 'eccentricity':
            difference = abs(oriIprops - nextIprops[i])/(float(max(oriIprops, nextIprops[i])) + 0.000000000001)
        elif distance_type == 'moments_hu':
            difference = np.abs(np.abs(oriIprops) - np.abs(np.array(nextIprops[i],dtype=np.float)))/np.abs( np.max(np.abs(np.vstack((oriIprops,nextIprops[i]))),axis=0) + 10**(-16)) 
        
        elif distance_type == 'equivalent_diameter':
            difference = abs(oriIprops - nextIprops[i])/(float(max(oriIprops, nextIprops[i])) + 0.000000000001)
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
   