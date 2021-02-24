import csv
import cv2
import os
import glob
import datetime
import math
import pandas
import argparse
import numpy as np
from tqdm import tqdm
from augvideo import aug_video
import pdb

def _id_adjust_kinetics(x):
    return 'X' + x + '.mp4'

def _id_adjust_ucf101(x):
    return x

def _get_class(name):
    clsname = name[2:]
    return clsname[0:clsname.find('_')]


# For a provided video, gets the farneback optical flow for each frame pair
# and maps out the magnitudes and angles throughout the clip in a histogram 
# and records the means, medians, and standard deviations of each bin
def dense(vid_path):
    cap = cv2.VideoCapture(cv2.samples.findFile(vid_path))
    success, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255
    mag_bins = []
    ang_bins = []
    success, frame2 = cap.read()
    while (success):
        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        mag_hist, _ = np.histogram(mag, bins=40, range=(0, 20))
        mag_bins.append(mag_hist.tolist())
        ang_hist, _ = np.histogram(ang, bins=8)
        ang_bins.append(ang_hist.tolist())
        prvs = next
        success, frame2 = cap.read()

    mag_mean = np.mean(mag_bins, axis=0)
    mag_median = np.median(mag_bins, axis=0)
    mag_std = np.std(mag_bins, axis=0)

    ang_mean = np.mean(ang_bins, axis=0)
    ang_median = np.median(ang_bins, axis=0)
    ang_std = np.std(ang_bins, axis=0)
    return [mag_mean, mag_median, mag_std, ang_mean, ang_median, ang_std]


def _groupings(mag_bins_df, ang_bins_df):
    groupings_mag_col = [
        [mag_bins_df.columns[0]],
        [mag_bins_df.columns[1]],
        [mag_bins_df.columns[2]],
        [mag_bins_df.columns[3]],
        [mag_bins_df.columns[4]],
        [mag_bins_df.columns[i] for i in range(5, 40)]]
    groupings_mag = {}
    grp_id = 0
    for grp_mag_col in groupings_mag_col:
        for col in grp_mag_col:
            groupings_mag[col.strip()] = grp_id
        grp_id += 1

    groupings_ang = {}
    for i in range(0, 8):
        groupings_ang[ang_bins_df.columns[i].strip()] = int(i / 2)
    return groupings_mag, groupings_ang

def calc_attributes(vids, dataset_path, mode):

    columns = np.zeros((len(vids), 144))
    headers = \
        [f", mag_mean_bin_{i}" for i in range(1, 41)] + \
        [f", mag_median_bin_{i}" for i in range(1, 41)] + \
        [f", mag_std_bin_{i}" for i in range(1, 41)] + \
        [f", ang_mean_bin_{i}" for i in range(1, 9)] + \
        [f", ang_median_bin_{i}" for i in range(1, 9)] + \
        [f", ang_std_bin_{i}" for i in range(1, 9)]

    for index, vid in enumerate(tqdm(vids)):
        try:
            results = dense(vid)
            columns[index, :] = np.concatenate(results)
        except Exception as e:
            print(e)
            print(f"Error reading {os.path.basename(vid)}. Skipping video")

    df = pandas.DataFrame(columns,
                          columns=headers)
    df.insert(0, "Video", [os.path.basename(vid) for vid in vids], True)
    df.set_index('Video')

    mag_bins_df = df[['Video'] + [i for i in df.columns][1:41]]
    mag_bins_df = mag_bins_df.set_index('Video')

    std_bins_df = df[['Video'] + [i for i in df.columns][81:121]]
    std_bins_df = std_bins_df.set_index('Video')

    ang_bins_df = df[['Video'] + [i for i in df.columns][121:129]]
    ang_bins_df = ang_bins_df.set_index('Video')

    mag_bins_df_max = mag_bins_df.idxmax(axis=1)
    ang_bins_df_max = ang_bins_df.idxmax(axis=1)

    std_a = np.asarray(std_bins_df)
    std_weight_sum = std_a[:, 0]
    for i in range(1, 40):
        std_weight_sum + ((40 - i) / 40.0) * std_a[:, i]
    std_boundaries = np.histogram(std_weight_sum, bins=5)[1]
    std_lookup = std_bins_df.index.tolist()

    groupings_mag, groupings_ang = _groupings(mag_bins_df, ang_bins_df)


    def _get_max_mag_bin(id_name):
        return groupings_mag[mag_bins_df_max.loc[id_adjust(os.path.basename(id_name))].strip()]


    def _get_max_ang_bin(id_name):
        return groupings_ang[ang_bins_df_max.loc[id_adjust(os.path.basename(id_name))].strip()]


    def _get_max_std_bin(id_name):
        id_name = os.path.basename(id_adjust(id_name))
        position = std_lookup.index(id_name)
        val = std_weight_sum[position]
        for i in range(std_boundaries.shape[0] - 1, 0, -1):
            if val >= std_boundaries[i]:
                return i
        return 0

    with open(os.path.join(dataset_path, mode+'_attribute_motion.csv'), 'w') as fp:
        for vid_id in mag_bins_df_max.keys():
            try:
                fp.write(f'{id_adjust(vid_id)},{_get_max_mag_bin(vid_id)},{_get_max_std_bin(vid_id)},{_get_max_ang_bin(vid_id)}\n')
            except:
                print("Error in calculating the attributes of the video " + vid_id)

    end_time = datetime.datetime.now()
    print("Finished!\nProcessing time: " + str(end_time - begin_time))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        help="Directory for where the video files are being held",
        default="../../Datasets/",
        dest="dir",
    )
    parser.add_argument(
        "--dst",
        help="Directory for where the generated video files are saved",
        default="../../Datasets/self_dataset/",
        dest="dst",
    )
    parser.add_argument(
        "--name",
        help="Name of dataset",
        default="UCF-101",
        dest="name",
    )
    parser.add_argument(
        "--ext",
        help="File extension for videos",
        default="avi",
        dest="ext",
    )
    parser.add_argument(
        "--csv",
        help="Name for csv file to save data to",
        default="optical_flow",
        dest="csv_name",
    )
    parser.add_argument(
        "--aug",
        help="Number of augmentation clips for each video",
        default=5,
        type=int,
        dest="aug_num",
    )
    args = parser.parse_args()
    
    if args.name == 'UCF-101':
        id_adjust = _id_adjust_ucf101
        pre_name = 'ucf101'
    else:
        id_adjust = _id_adjust_kinetics
        pre_name = 'kinetics'

    begin_time = datetime.datetime.now()
    # load test video list in the dataset
    dataset_path = os.path.join(args.dir, args.name)
    split_path = os.path.join(args.dir, 'TA2_splits')
    known_list = os.path.join(split_path, '{}_train_knowns_revised.csv'.format(pre_name))
    unknown_list = os.path.join(split_path, '{}_train_unknowns_revised.csv'.format(pre_name))
    # load test videos
    cnt = 0
    known_video = []
    f = csv.reader(open(known_list,'r'))
    for line in f:
        if args.name == 'UCF-101':
            cur_video = line[0].split('/')[1]
        else:
            cur_video = "X"+line[0]+".mp4"
        cur_index = line[1]
        video_path = os.path.join(dataset_path, cur_video)
        if cnt%10 == 0 and os.path.exists(video_path):
            known_video.append(video_path)
        cnt += 1
    unknown_video = []
    f = csv.reader(open(unknown_list,'r'))
    for line in f:
        if args.name == 'UCF-101':
            cur_video = line[0].split('/')[1]
        else:
            cur_video = "X"+line[0]+".mp4"
        cur_index = line[1]
        video_path = os.path.join(dataset_path, cur_video)
        if os.path.exists(video_path):
            unknown_video.append(video_path)
    print(f"Original video count: known {len(known_video)}/ unknown {len(unknown_video)}")

    # generate new videos
    aug_known_video = aug_video(known_video, args.dst, 'known', args.aug_num)
    aug_unknown_video = aug_video(unknown_video, args.dst, 'unknown', args.aug_num)

    # calculate representation attributes
    print("Calculating the attributes of known videos...")
    calc_attributes(aug_known_video, args.dir, 'known')
    print("Calculating the attributes of unknown videos...")
    calc_attributes(aug_unknown_video, args.dir, 'unknown')
