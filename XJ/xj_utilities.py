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
            patch_loc_plot = rescale_grayscale_image(patch_loc_plot)
        except:
            sys.stderr('Cannot rescale the grayscale image. Check if function rescale_grayscale_image is available.')
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
        fig_dismap = plt.imshow(distance_im, cmap=plt.cm.gray)
        fig_stat = plt.figure()
        fig_stat = plt.scatter(dis_hist_bins_mid,np.log10(distance_im_his[0]));
        fig_stat = plt.hold
        fig_stat = plt.plot(dis_hist_bins_mid[1:],dis_hist_LF.slope * dis_hist_bins_mid[1:] + dis_hist_LF.intercept,color='red');
        fig_stat = plt.title('slope = '+str(dis_hist_LF.slope)+'  Max_dist = '+str(dis_hist_max) + '\n  1stBinRatio = ' + str(dis_hist_1st_bin_NumRatio)+ '  LastBinAreaRatio = ' + str(dis_hist_last_bin_AreaRatio))
        fig_stat = plt.xlabel('Distance')
        fig_stat = plt.ylabel('Log10(#Pixel)')
    return features_DT, distance_im

### Image Processing ### 
def fun_rescale_grayscale_image(inputdata,invcol=False):
    inputdata = np.array(inputdata);
    vmin = np.percentile(inputdata, 0);
    vmax = np.percentile(inputdata, 100);
    rescale_image = (inputdata - vmin ) / (vmax - vmin);
    rescale_image = np.maximum(np.minimum(rescale_image, 1), 0)
    rescale_image = skimage.img_as_ubyte(rescale_image)
    if invcol:
        rescale_image = 255 - rescale_image
    return rescale_image
