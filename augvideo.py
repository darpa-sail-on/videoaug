import os
import cv2
import random
from PIL import Image
import vidaug.augmentors as va
import numpy as np
import pdb

def _brightness(image, random_br):
    '''
    Randomly changes the brightness of the input image.

    Protected against overflow.
    '''
    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    #To protect against overflow: Calculate a mask for all pixels
    #where adjustment of the brightness would exceed the maximum
    #brightness value and set the value to the maximum at those pixels.
    mask = hsv[:,:,2] * random_br > 255
    v_channel = np.where(mask, 255, hsv[:,:,2] * random_br)
    hsv[:,:,2] = v_channel

    return cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR) 

def video_loader(seq_path, mode):
    frames = []
    fps = 0
    random_br = np.random.uniform(0.5,2.0)
    if os.path.exists(seq_path):
        cap = cv2.VideoCapture(seq_path)
        fps = cap.get(cv2.CAP_PROP_FPS) # get the frame rate of the video
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == False:
                break
            #rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if mode == 'unknown':
                frame = _brightness(frame, random_br)
            frames.append(frame)
        cap.release()
    else:
        print('{} does not exist'.format(seq_path))

    return frames, fps

def aug_video(video_list, save_path, mode, aug_num):
    new_video_list = []
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    # initialize video writer
    fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')

    # generate new videos
    for k in range(len(video_list)):
        frames, fps = video_loader(video_list[k], mode)
        for j in range(aug_num):
            # video agumentation steps
            sometimes = lambda aug: va.Sometimes(1, aug) # used to apply augmentor with 100% probability
            ran_num = int(random.randint(1,4)) # select random number of steps
            aug_seq = va.SomeOf([
            sometimes(va.HorizontalFlip()), # horizontally flip the video with 100% probability
                va.RandomCrop((192, 256)), # extract random crop of the video (crop_height, crop_width)
                va.Downsample(0.8), # temporally downsample a video by deleting some of its frames
                va.Upsample(1.5), # temporally upsampling a video by deleting some of its frames
            ], ran_num, random_order=False)
            # video generation
            aug_frames, aug_steps = aug_seq(frames)
            height, width, _ = aug_frames[0].shape
            cur_video = video_list[k].split('/')
            video_filename = os.path.join(save_path, "".join([str(x) for x in aug_steps])+cur_video[-1])
            # save videos
            out = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))
            for i in range(len(aug_frames)):
                out.write(aug_frames[i])
            # close out the video writer
            out.release()
            new_video_list.append(video_filename)

    return new_video_list
