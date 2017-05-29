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

# Function to find the threshold by gradient alignemnt

# Function to find the threshold by gradient alignemnt

def fun_threshold_gradAlig(image,scanrange=(160,240),step=5,o_size_gaussfilt=1,show_score_plot=True,method='tot_dot_pdt'):
    """
        image: a grayscale image(0-255)
        step: step for grayscale scan. can be integer number larger than 0
        o_size_gaussfilt: size of the gaussian filter. 
                          Not apply the gaussian filter to the image if equals 0
        method: (1)tot_dot_pdt; (2)avg_cos; (3) avg_dot_pdt
        package needed: scipy.ndimage as ndi; numpy as np; matplotlib.pyplot as plt;  
        
    """
    if o_size_gaussfilt > 0:
        tempImage = ndi.filters.gaussian_filter(image, o_gass_filt_sigma)
    else:
        tempImage = image
    gradX, gradY = np.gradient(tempImage)
    image_size = image.size
    temp_TH_aggrement = {}
    
    if method == 'tot_dot_pdt':
        grad_list = np.concatenate([gradX.flatten(),gradY.flatten()])
        for tempTH in np.arange(scanrange[0],scanrange[1],step):
            tempGx, tempGy = np.gradient(tempImage < tempTH)
            temp_grad_list = np.concatenate([tempGx.flatten(), tempGy.flatten()])
            temp_grad_aggrement_score = np.dot(temp_grad_list, grad_list)
    #         temp_grad_aggrement_score = np.dot(tempGx.flatten(),gradX_gf.flatten()) + np.dot(tempGy.flatten(), gradY_gf.flatten())
            temp_TH_aggrement[tempTH] = temp_grad_aggrement_score  
    
    if method == 'avg_dot_pdt':
        for tempTH in np.arange(scanrange[0],scanrange[1],step):
            tempGx, tempGy = np.gradient(tempImage < tempTH)
            non_zero =np.logical_and(np.logical_or(tempGx != 0,tempGy != 0),np.logical_or(gradX !=0, gradY !=0 ))
            temp_TH_aggrement[tempTH] = (tempGx[non_zero]*gradX[non_zero] + tempGy[non_zero]*gradY[non_zero]).mean()       
        
    if method == 'avg_cos':
        grad_list_gf = np.transpose(np.stack([gradX.flatten(),gradY.flatten()]))
        for tempTH in np.arange(scanrange[0],scanrange[1],step):
            tempGx, tempGy = np.gradient(tempImage < tempTH)
            
            tempG_list = np.transpose(np.stack([tempGx.flatten(),tempGy.flatten()])) # gradient vector of each pixel on BW image
            non_zero_G = np.logical_or(tempGx > 0, tempGy > 0).flatten()
            temp_n_nonzero = np.count_nonzero(non_zero_G) # num of nonzero gradient point on BW image
            temp_n_nonzero_count = 0 
            if temp_n_nonzero > 0:
                temp_total_score = 0
                for tempIndex in non_zero_G.nonzero()[0]:
                    temp_doc_product = grad_list_gf[tempIndex][0]*tempG_list[tempIndex][0] + grad_list_gf[tempIndex][1]*tempG_list[tempIndex][1]
                    temp_norm_grayscale = (grad_list_gf[tempIndex][0]**2 + grad_list_gf[tempIndex][1]**2)**0.5
                    temp_norm_BW = (tempG_list[tempIndex][0]**2 + tempG_list[tempIndex][1]**2)**0.5
                    if temp_norm_grayscale * temp_norm_BW > 0:
                        temp_total_score = temp_total_score + temp_doc_product/float(temp_norm_grayscale * temp_norm_BW)
                        temp_n_nonzero_count = temp_n_nonzero_count + 1
                    else:
                        continue
        #                 print(temp_norm_grayscale,temp_norm_BW)
                if temp_n_nonzero_count > 0:
                    temp_score = temp_total_score/float(temp_n_nonzero_count)
        #             print(tempTH,temp_score,temp_n_nonzero_count,temp_n_nonzero)
                else:
                    print(tempTH)
                    temp_score = 0
            else:
                temp_score = 0
            temp_TH_aggrement[tempTH] = temp_score   
    
    tempKeys, tempValues = zip(*sorted(temp_TH_aggrement.items()))
    grad_threshold = tempKeys[np.argmax(tempValues)]
    
    if show_score_plot==True:
        fig_grad_THscan_score = plt.figure()
        fig_grad_THscan_score = plt.plot(tempKeys, tempValues)
        fig_grad_THscan_score = plt.xlabel('Grayscale threshold')
        fig_grad_THscan_score = plt.ylabel('Score')
        fig_grad_THscan_score = plt.title('Score method used: '+method+'\nBest threshold = %s' % grad_threshold)
        fig_grad_THscan_score = plt.grid(True)
    return grad_threshold