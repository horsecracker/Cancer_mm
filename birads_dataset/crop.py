from PIL import Image
import os
import numpy as np

from skimage import io
import scipy.ndimage


def crop_to_max_square(img_file, outfile, target_square_length):
    im = Image.open(img_file)

    input_width, input_height = im.size
    crop_rect_length = max(input_width, input_height)
    crop_rect = (max([input_width-crop_rect_length,0])/2, max([input_height-crop_rect_length, 0])/2, crop_rect_length, crop_rect_length)

    im = im.crop(crop_rect)
    target_size = (target_square_length, target_square_length)
    im.thumbnail(target_size)
    im.convert("RGB").save(outfile, "JPEG")


def crop_square(img_file, outfile, target_square_length):
    im_array = scipy.ndimage.imread(img_file, mode='L')

    pix_array=np.zeros((im_array.shape[0],im_array.shape[1]) )
    pix_array[np.where(im_array > 0.05)] = 255

    idx_y =np.where(pix_array.any(axis=0))[0]   # remove the black clomn strip 
    miny,maxy = seg_array(idx_y)
    idx_x = np.where(pix_array[:,miny:maxy].any(axis=1))[0]  # remove 
    minx,maxx = seg_array(idx_x)                             # breast border along x    # get the squre boxes

    test = im_array[minx:maxx,miny:maxy]
    im_temp = Image.fromarray(test)
    target_size = (target_square_length, target_square_length)
    im_temp.thumbnail(target_size)
    im.convert("RGB").save(outfile+'_0', "JPEG")

    

    miny = max(0, int(miny*0.6))                # find breast the border along y    
    maxy = min(int(maxy*1.4), im_array.shape[1])
    minx = max(0, int(minx*0.9))                # find breast the border along y    
    maxx = min(int(maxx*1.1), im_array.shape[0])


    minx, maxx, miny, maxy = box_square(pix_array,minx, maxx, miny, maxy)
    newimg=im_array[minx:maxx,miny:maxy]
    im = Image.fromarray(newimg)
    target_size = (target_square_length, target_square_length)
    im.thumbnail(target_size)
    im.convert("RGB").save(outfile, "JPEG")

'''
def box_square(pix_array, minx, maxx, miny, maxy):
    # input: bounding of the breast
    # output: square box out of 10 random boxi choosing the one has most content
    lenx = maxx-minx
    leny = maxy-miny
    max_distortion = 1.2
    if float(1/max_distortion) < lenx/leny < max_distortion:
        return minx, maxx, miny, maxy
    brightness=0
    n = 10
    if float(lenx)/float(leny) >= max_distortion:
    #if maxx-minx > maxy-miny:
        imgsize = int(leny*max_distortion)
        for i in range(n):
            xstart = np.random.randint(minx, maxx-imgsize)
            bright = np.mean(pix_array[xstart:xstart+imgsize, miny:maxy])
            if bright > brightness:
                brightness = bright
                newx = xstart
        return newx, newx+imgsize, miny, maxy
    if float(leny)/float(lenx) >= max_distortion:
    #if maxx-minx <= maxy-miny:
        imgsize = int(lenx*max_distortion)
        for i in range(n):
            ystart = np.random.randint(miny, maxy-imgsize)
            bright = np.mean(pix_array[minx:maxx, ystart:ystart+imgsize])
            if bright > brightness:
                brightness = bright
                newy = ystart
        return minx, maxx, newy, newy+imgsize
'''

def box_square(pix_array, minx, maxx, miny, maxy):
    # input: bounding of the breast
    # output: square box out of 10 random boxi choosing the one has most content
    if maxx-minx == maxy-miny:
        return minx, maxx, miny, maxy
    brightness=0
    n = 20
    if maxx-minx > maxy-miny:
        imgsize = maxy-miny
        for i in range(n):
            xstart = np.random.randint(minx, maxx-maxy+miny)
            bright = np.mean(pix_array[xstart:xstart+imgsize, miny:maxy])
            if bright > brightness:
                brightness = bright
                newx = xstart
        return newx, newx+imgsize, miny, maxy
    if maxx-minx <= maxy-miny:
        imgsize = maxx-minx
        for i in range(n):
            ystart = np.random.randint(miny, maxy-maxx+minx)
            bright = np.mean(pix_array[minx: maxx, ystart: ystart+imgsize])
            if bright > brightness:
                brightness = bright
                newy = ystart
        return minx, maxx, newy, newy+imgsize
        
def seg_array(idx_list):
    # input: idx of the pixel strips where there is nonzero values
    # output: find the biggest pix section
    seg = []
    start= begin = 0
    end = stop = 0
    for i in range(1, len(idx_list)):
        if idx_list[i]== idx_list[i-1] +1 and i!=len(idx_list)-1:
            stop = i
        else:
            if (stop-begin)>(end-start):
                start= begin
                end= stop
            begin = stop = i        
    return idx_list[start], idx_list[end]


def crop_all_images_to_max_square(image_dir, target_size):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    full_dir = os.path.join(dir_path, image_dir)

    for root, dirs, files in os.walk(full_dir):
        for name in files:
            if name.split('.')[-1] in {'jpg', 'png'}:
                full_path = os.path.join(root, name)
                parts = [el for el in full_path.split('/') if el]
                target_folder_path = '/{}-sq-{}/{}/'.format('/'.join(parts[:-2]), target_size, parts[-2])
                target_full_path = '{}{}.jpg'.format(target_folder_path, parts[-1].split('.')[0])
                if not os.path.exists(target_folder_path):
                    os.makedirs(target_folder_path)
                #crop_to_max_square(full_path, target_full_path, target_size)
                crop_square(full_path, target_full_path, target_size)


crop_all_images_to_max_square('train', 256)
crop_all_images_to_max_square('dev', 256)


#crop_all_images_to_max_square('train', 299)
#crop_all_images_to_max_square('dev', 299)
