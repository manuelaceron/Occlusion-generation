import os
# Inspiration: https://github.com/honnibal/spacy-ray/pull/
# 1/files#diff-7ede881ddc3e8456b320afb958362b2aR12-R45

import cv2





def get_srcNmask(image_file,img_path,mask_path):
    """
    Get the face image and mask
    """
    img_name=image_file.split(".")[0]
    src_img= cv2.imread(os.path.abspath(os.path.join(img_path,image_file)),-1)
    
    #src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)

    src_mask= cv2.imread(os.path.join(mask_path, f"{img_name}.png"))
    
    #src_mask=cv2.resize(src_mask,(1024,1024),interpolation= cv2.INTER_LANCZOS4)
    src_mask=cv2.cvtColor(src_mask,cv2.COLOR_RGB2GRAY)

    return src_img, src_mask

def get_occluderNmask(occluder_file,img_path,mask_path):
    occluder_name=occluder_file.split(".")[0]
    ori_occluder_img= cv2.imread(os.path.abspath(os.path.join(img_path,occluder_file)),-1)#cv2.IMREAD_UNCHANGED)#
    
    if ori_occluder_img.shape[2] < 4:
        print('no alpha channel', occluder_name)
        return
       
    occluder_mask= cv2.imread(os.path.abspath(os.path.join(mask_path,occluder_name+".png")))
    occluder_mask = cv2.cvtColor(occluder_mask, cv2.COLOR_BGR2GRAY)
    (thresh, occluder_mask) = cv2.threshold(occluder_mask, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) #new
    
    h, w, d = ori_occluder_img.shape
    
    #Make img ans mask same size: mask is smaller
    ori_occluder_img = cv2.resize(ori_occluder_img,(occluder_mask.shape[1],occluder_mask.shape[0]),interpolation= cv2.INTER_LANCZOS4) 
    #print(ori_occluder_img.shape, occluder_mask.shape)
    
    
    #cropped out the hand img
    
    try:
        occluder_img=cv2.bitwise_and(ori_occluder_img,ori_occluder_img,mask=occluder_mask)
    except Exception as e:
        print(e)
        return

    occluder_rect = cv2.boundingRect(occluder_mask)
    cropped_occluder_mask = occluder_mask[ occluder_rect[1]:(occluder_rect[1]+occluder_rect[3]),occluder_rect[0]:(occluder_rect[0]+occluder_rect[2])]
    cropped_occluder_img = occluder_img[ occluder_rect[1]:(occluder_rect[1]+occluder_rect[3]),occluder_rect[0]:(occluder_rect[0]+occluder_rect[2])] 
    return cropped_occluder_img, cropped_occluder_mask    