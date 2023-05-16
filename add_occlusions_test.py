import os
import numpy as np
import cv2
from utils.random_shape_generator import *
from utils.utils import *
from utils.paste_over import *
import random
import skimage
import glob
import sys
import pdb

def set_random_seed(seed, deterministic=False):
    """Set random seed.
    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    # ia.seed(seed)
    
    random.seed(seed)
    np.random.seed(seed)
    

# Generate a random occlusion (img and mask) with a given random texture:
def get_randomOccluderNmask(occlude_dir):
    #get random shape mask
    rad = np.random.rand()
    edgy = np.random.rand()
    mask_shape=512 #determine this...

    # This I would need for shades and fabrics, for shades, rectangular, for fabric, covering the building...
    # consider skimage.draw.polygon(r, c, shape=None)
    no_of_points=random.randint(3, 15)
    a = get_random_points(n=no_of_points, scale=mask_shape) 
    x,y, _ = get_bezier_curve(a,rad=rad, edgy=edgy)
    occluder_mask=skimage.draw.polygon2mask((mask_shape,mask_shape),list(zip(x,y))).astype(np.uint8)*255

    # get random texture: use this for fabrics...
    #texture_list= os.listdir(occlude_dir)
    # texture_list.remove('freckled')
    #texture_choice=random.sample(texture_list,1)[0]
    texture_choice = "meshed"
    texture_occlude_dir = os.path.join(occlude_dir, texture_choice)
    texture_img = random.sample(glob.glob(f"{texture_occlude_dir}/*.jpg"),1)[0]
    ori_occluder_img= cv2.imread(texture_img,-1)
    ori_occluder_img=cv2.resize(ori_occluder_img,(mask_shape,mask_shape))
    try:
        ori_occluder_img = cv2.cvtColor(ori_occluder_img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(e)
    #cropped out the hand img
    try:
        # get texture to the polygon mask
        occluder_img=cv2.bitwise_and(ori_occluder_img,ori_occluder_img,mask=occluder_mask)
    except Exception as e:
        print(e)
        return
    occluder_rect = cv2.boundingRect(occluder_mask)
    cropped_occluder_mask = occluder_mask[ occluder_rect[1]:(occluder_rect[1]+occluder_rect[3]),occluder_rect[0]:(occluder_rect[0]+occluder_rect[2])]
    cropped_occluder_img = occluder_img[ occluder_rect[1]:(occluder_rect[1]+occluder_rect[3]),occluder_rect[0]:(occluder_rect[0]+occluder_rect[2])] 
    return cropped_occluder_img, cropped_occluder_mask

def resize(occluder_img, occluder_mask,factor,occ_w,occ_h):
    dim = (int(occ_w*factor), int(occ_h*factor))
    occluder_img = cv2. resize(occluder_img, dim)
    occluder_mask = cv2. resize(occluder_mask, dim)
    return occluder_img, occluder_mask

def occlude_images(img_file, img_path, mask_path, oc_file, oc_img_path, oc_mask_path):

    oc_type = oc_file.split('_')[0]
    
    # get source img and mask
    src_img, src_mask = get_srcNmask(img_file,img_path,mask_path)

    #get occluder img and mask (given a texture)
    occluder_img , occluder_mask = get_occluderNmask(oc_file,oc_img_path,oc_mask_path)

    src_h, src_w, src_d = src_img.shape
    occ_h, occ_w, occ_d = occluder_img.shape

    src_rect = cv2.boundingRect(src_mask) # rectangle around mask
    x,y,w,h = src_rect
    height, width = src_mask.shape
    #img= cv2.rectangle(src_img,(x,y),(x+w,y+h),(0,255,0),2)

    #------- Location constraints-----------#

    in_floor = {"truck", "car"}
    if oc_type in  in_floor:
        occluder_coord = np.random.uniform([x,y+1.2*h], [x+w,height]) 
        
    elif oc_type == "treet":
        occluder_coord = np.random.uniform([x,y+0.8*h], [x+w,height]) 
    
    elif oc_type == "ppl":
        occluder_coord = np.random.uniform([x,y+1.1*h], [x+w,height]) 
    
    elif oc_type == "lamp" or oc_type == "sign":
        occluder_coord = np.random.uniform([x,y+0.60*h], [x+w,height]) 
    
    elif oc_type == "elec":
        occluder_coord = np.random.uniform([x,y+0.5*h], [x+w,0.8*h]) 

    else:
        occluder_coord = np.random.uniform([x,y], [x+w,y+h]) #random

    
    #------- Size constraints-----------#

    if  oc_type == "car":
        factor = np.random.uniform((src_w*1.5)/occ_w, (src_w*0.85)/occ_w) 
        occluder_img, occluder_mask = resize(occluder_img, occluder_mask, factor,occ_w,occ_h)

    elif oc_type == "truck":        
        factor = np.random.uniform((src_w*2)/occ_w, (src_w*0.85)/occ_w) 
        occluder_img, occluder_mask = resize(occluder_img, occluder_mask, factor,occ_w,occ_h)
    
    elif oc_type == "treet":
        factor = np.random.uniform((src_h)/occ_h, (src_h*0.75)/occ_h) 
        occluder_img, occluder_mask = resize(occluder_img, occluder_mask, factor,occ_w,occ_h)
    
    elif oc_type == "flag":
        factor = np.random.uniform((src_w*0.5)/occ_w, (src_w*0.25)/occ_w) 
        occluder_img, occluder_mask = resize(occluder_img, occluder_mask, factor,occ_w,occ_h)
    
    elif oc_type == "lamp" or oc_type == "sign":
        factor = np.random.uniform((src_h)/occ_h, (src_h*0.8)/occ_h) 
        occluder_img, occluder_mask = resize(occluder_img, occluder_mask, factor,occ_w,occ_h)
    
    elif oc_type == "ppl":
        factor = np.random.uniform((src_h*0.5)/occ_h, (src_h*0.35)/occ_h) 
        occluder_img, occluder_mask = resize(occluder_img, occluder_mask, factor,occ_w,occ_h)
    
        
    occlusion_mask=np.zeros(src_mask.shape, np.uint8)
    occlusion_mask[(occlusion_mask>0) & (occlusion_mask<255)]=255

    #paste occluder to src image

    gsr = 1 #radius for Gaussian blus on alpha channel (mask)
    gsr_e = 1 #radius for Gaussian blus on edgeds on occluded image
    result_img, result_mask, occlusion_mask= paste_over(occluder_img,occluder_mask,src_img,src_mask,occluder_coord,occlusion_mask,False, gsr, oc_type)
   

    #blur edges of occluder
    kernel = np.ones((3,3),np.uint8) #was 5
    occlusion_mask_edges=cv2.dilate(occlusion_mask,kernel,iterations = 2)-cv2.erode(occlusion_mask,kernel,iterations = 2)
    ret, filtered_occlusion_mask_edges = cv2.threshold(occlusion_mask_edges, 240, 255, cv2.THRESH_BINARY)
    blurred_image = cv2.GaussianBlur(result_img,(gsr_e,gsr_e),0)
    result_img = np.where(np.dstack((np.invert(filtered_occlusion_mask_edges==255),)*3), result_img, blurred_image)

    #save images
    save_images(img_file,result_img,result_mask,occlusion_mask, gsr, gsr_e)

    #cv2.imwrite(os.path.join(outputMaskDir, f"{img_file}.png"),result_mask)

    

def save_images(img_name,image,mask,occlusion_mask,src_img, src_mask):
    if not os.path.exists(outputImgDir):
        os.makedirs(outputImgDir)
    if not os.path.exists(outputMaskDir):
        os.makedirs(outputMaskDir)
    if not os.path.exists(occlusionMaskDir):
        os.makedirs(occlusionMaskDir)
    if not os.path.exists(output_ori_img_mask):
        os.makedirs(output_ori_img_mask)
     
    cv2.imwrite(os.path.join(outputImgDir, img_name),image) #
    cv2.imwrite(os.path.join(outputMaskDir, img_name.split('.')[0]+".png"),mask)
    cv2.imwrite(os.path.join(occlusionMaskDir, img_name.split('.')[0]+".png"),occlusion_mask)

    #cv2.imwrite(os.path.join(output_ori_img_dir, img_name),src_img)
    #cv2.imwrite(os.path.join(output_ori_img_mask, img_name.split('.')[0]+".png"),src_mask)


img_file= "monge_97.jpg"


#oc_file="tree_3.png"
oc_img_path= "/home/cero_ma/MCV/add occlusions/occluders/images/"
oc_mask_path="/home/cero_ma/MCV/add occlusions/occluders/labels/"





import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-m", "--mode", default="train")
args = parser.parse_args()
mode = args.mode
print('Mode: ',mode)


base_in_dir = "/home/cero_ma/MCV/window_benchmarks/originals/split_data/ecp-refined-mcv/"
img_path= os.path.join(base_in_dir, mode, "images")
mask_path= os.path.join(base_in_dir, mode, "labels")

base_out_dir = "/home/cero_ma/MCV/window_benchmarks/originals/split_data/ecp-refmcv-occluded60/"
outputImgDir = os.path.join(base_out_dir, mode, "images")
outputMaskDir = os.path.join(base_out_dir, mode, "occ_labels")
occlusionMaskDir = os.path.join(base_out_dir, mode, "occ_masks")
#output_ori_img_dir = os.path.join(base_out_dir, mode, "clean_images")
output_ori_img_mask = os.path.join(base_out_dir, mode, "labels")

#set seed
set_random_seed(112) #ecp-refmcv-occluded60: 112

oc_files = os.listdir(oc_img_path)


im_files = os.listdir(img_path)
random.shuffle(im_files)
oc_ratio = int(len(im_files)*0.6)
im_sample = random.sample(im_files, oc_ratio) #Facade images to occlude
print("Total ", len(im_files), " files, occlusion ", len(im_sample), " samples")
#
# Take len(im_sample)

#Half of the occlusions are vegetation
occ_vegetation = [x for x in oc_files if x.split('_')[0] in {'treet'}]
occ_vegetation_up = [x for x in oc_files if x.split('_')[0] in {'tree'}]
occ_any = [x for x in oc_files if x.split('_')[0] not in {'tree', 'treet'}]

occ_veg_total = len(occ_vegetation) + len(occ_vegetation_up) #Total veg dataset occluders

random.shuffle(occ_vegetation)
random.shuffle(occ_vegetation_up)
random.shuffle(occ_any)

any_occ_size = int(len(im_sample)*0.5)


veg_occ_size_up = int((len(im_sample) - any_occ_size)*0.4)

veg_occ_size = (len(im_sample) - any_occ_size - veg_occ_size_up)
veg_occ_size_total = veg_occ_size + veg_occ_size_up


veg_file = []

if occ_veg_total < veg_occ_size: # If there are less occluders than needed
    veg_file = random.sample(occ_vegetation_up, len(occ_vegetation_up)) # Take all up trees
    veg_file = veg_file + random.sample(occ_vegetation, len(occ_vegetation)) # Take all stand trees

    veg_file = veg_file + random.choices(occ_vegetation + occ_vegetation_up, k= (veg_occ_size_total- occ_veg_total) )
else:
   
    
    if len(occ_vegetation) < veg_occ_size:
        veg_file = veg_file + random.sample(occ_vegetation, len(occ_vegetation))
        veg_file = veg_file + random.choices(occ_vegetation, k= (veg_occ_size - len(occ_vegetation)))
    else:
        veg_file = veg_file + random.sample(occ_vegetation, veg_occ_size)
    

    if len(occ_vegetation_up) < veg_occ_size_up:
        veg_file = veg_file + random.sample(occ_vegetation_up, len(occ_vegetation_up))
        veg_file = veg_file + random.choices(occ_vegetation_up, k= (veg_occ_size_up - len(occ_vegetation_up)))
    else:
        veg_file = veg_file + random.sample(occ_vegetation_up, veg_occ_size_up)


if len(occ_any) < any_occ_size:
    any_file = random.sample(occ_any, len(occ_any))
    any_file = any_file + random.choices(occ_any, k = (any_occ_size - len(occ_any)))
else:
    any_file = random.sample(occ_any, any_occ_size)





woc_file = veg_file + any_file
random.shuffle(woc_file)

#woc_file = random.choices(oc_files, weights=test_prio, k = len(im_sample)) #with replacement?
print('Total occlusions: ', woc_file)

count = 0
countup = 0
for img_file in im_sample:
    try:
        oc_file = woc_file.pop() 
        if oc_file.split('_')[0] in {'treet'}:
            count = count + 1
        if oc_file.split('_')[0] in {'tree'}:
            countup = countup + 1
            
        occlude_images(img_file, img_path, mask_path, oc_file, oc_img_path, oc_mask_path)
    except Exception as e:
        print(e)
        print(f'Failed: {img_file} , {oc_file}')
        continue
print('Total vegetation occluders: ',count, 'and up ', countup)

# Images in im_files - im_sample just copy...

for img_file in im_files:
    if not img_file in im_sample:
        cmd1 = 'cp -r ' + ' ' + os.path.join(img_path,img_file) + ' ' +  os.path.join(outputImgDir,img_file)
        cmd2 = 'cp -r ' + ' ' + os.path.join(mask_path,img_file.split('.')[0]+'.png') + ' ' + os.path.join(outputMaskDir,img_file.split('.')[0]+'.png')
        
        # TODO: What should go in occlabels dir? This is the occlusion mask

        im = cv2.imread(os.path.join(img_path,img_file))
        im.shape[0:2]
        black_img = np.zeros(im.shape[0:2], dtype = np.uint8)
        cv2.imwrite(os.path.join(occlusionMaskDir, img_file.split('.')[0]+'.png'),black_img)
        #cmd3 = 'cp -r ' + ' ' + os.path.join(mask_path,img_file.split('.')[0]+'.png') + ' ' + os.path.join(occlusionMaskDir,img_file.split('.')[0]+'.png')

        os.system(cmd1)
        os.system(cmd2)
        #os.system(cmd3)
    # Fill folder with clean (original) labels 
    cmd4 = 'cp -r ' + ' ' + os.path.join(mask_path,img_file.split('.')[0]+'.png') + ' ' + os.path.join(output_ori_img_mask,img_file.split('.')[0]+'.png')
    os.system(cmd4)

""" for oc_file in oc_files:
    try:
        occlude_images(img_file, img_path, mask_path, oc_file, oc_img_path, oc_mask_path)
    except Exception as e:
        print(e)
        print(f'Failed: {img_file} , {oc_file}')
        continue
     """