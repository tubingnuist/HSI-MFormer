import json
import math
import pandas as pd
import torch
import os
import sys
import shutil
import numpy as np
import glob
import random

def image2cols(image,patch_size,stride):
    """
    image:需要切分为图像块的图像
    patch_size:图像块的尺寸，如:(10,10)
    stride:切分图像块时移动过得步长，如:5
    """
    import numpy as np
    if len(image.shape) == 2:
        # 灰度图像
        imhigh,imwidth = image.shape
    if len(image.shape) == 3:
        # RGB图像
        imhigh,imwidth,imch = image.shape
    ## 构建图像块的索引
    range_y = np.arange(0,imhigh - patch_size[0],stride)
    range_x = np.arange(0,imwidth - patch_size[1],stride)
    if range_y[-1] != imhigh - patch_size[0]:
        range_y = np.append(range_y,imhigh - patch_size[0])
    if range_x[-1] != imwidth - patch_size[1]:
        range_x = np.append(range_x,imwidth - patch_size[1])
    sz = len(range_y) * len(range_x)  ## 图像块的数量
    if len(image.shape) == 2:
        ## 初始化灰度图像
        res = np.zeros((sz,patch_size[0],patch_size[1]))
    if len(image.shape) == 3:
        ## 初始化RGB图像
        res = np.zeros((sz,patch_size[0],patch_size[1],imch))
    index = 0
    for y in range_y:
        for x in range_x:
            patch = image[y:y+patch_size[0],x:x+patch_size[1]]
            res[index] = patch
            index = index + 1
    return res

def col2image(coldata,imsize,stride):
    """
    coldata: 使用image2cols得到的数据
    imsize:原始图像的宽和高，如(321, 481)
    stride:图像切分时的步长，如10
    """
    patch_size = coldata.shape[1:3]
    if len(coldata.shape) == 3:
        ## 初始化灰度图像
        res = np.zeros((imsize[0],imsize[1]))
        w = np.zeros(((imsize[0],imsize[1])))
    if len(coldata.shape) == 4:
        ## 初始化RGB图像
        res = np.zeros((imsize[0],imsize[1],3))
        w = np.zeros(((imsize[0],imsize[1],3)))
    range_y = np.arange(0,imsize[0] - patch_size[0],stride)
    range_x = np.arange(0,imsize[1] - patch_size[1],stride)
    if range_y[-1] != imsize[0] - patch_size[0]:
        range_y = np.append(range_y,imsize[0] - patch_size[0])
    if range_x[-1] != imsize[1] - patch_size[1]:
        range_x = np.append(range_x,imsize[1] - patch_size[1])
    index = 0
    for y in range_y:
        for x in range_x:
            res[y:y+patch_size[0],x:x+patch_size[1]] = res[y:y+patch_size[0],x:x+patch_size[1]] + coldata[index]
            w[y:y+patch_size[0],x:x+patch_size[1]] = w[y:y+patch_size[0],x:x+patch_size[1]] + 1
            index = index + 1
    return res / w

def image2cols_batch(image,patch_size,stride):
    batch = image.shape[0]
    sz = (image.shape[1]/patch_size[0]) * (image.shape[2]/patch_size[1])
    Res = np.zeros((batch,int(sz),patch_size[0],patch_size[1],3))
    for i in range(batch):
        current_image = image[i]
        Res[i] = image2cols(current_image,patch_size,stride)
    return Res

def col2image_batch(coldata,imsize,stride):
    batch = coldata.shape[0]
    final_image = np.zeros((batch,imsize[0],imsize[1],3))
    for i in range(batch):
        final_image[i] = col2image(coldata[i],imsize,stride)
    return final_image

def adjust_learning_rate(optimizer, epoch, init_param_lr, lr_epoch_1, lr_epoch_2):
    i = 0
    for param_group in optimizer.param_groups:
        init_lr = init_param_lr[i]
        i += 1
        if(epoch <= lr_epoch_1):
            param_group['lr'] = init_lr * 0.1 ** 0
        elif(epoch <= lr_epoch_2):
            param_group['lr'] = init_lr * 0.1 ** 1
        else:
            param_group['lr'] = init_lr * 0.1 ** 2

import matplotlib.pyplot as plt
def draw_roc(frr_list, far_list, roc_auc):
    plt.switch_backend('agg')
    plt.rcParams['figure.figsize'] = (6.0, 6.0)
    plt.title('ROC')
    plt.plot(far_list, frr_list, 'b', label='AUC = %0.4f' % roc_auc)
    plt.legend(loc='upper right')
    plt.plot([0, 1], [1, 0], 'r--')
    plt.grid(ls='--')
    plt.ylabel('False Negative Rate')
    plt.xlabel('False Positive Rate')
    save_dir = './save_results/ROC/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig('./save_results/ROC/ROC.png')
    file = open('./save_results/ROC/FAR_FRR.txt', 'w')
    save_json = []
    dict = {}
    dict['FAR'] = far_list
    dict['FRR'] = frr_list
    save_json.append(dict)
    json.dump(save_json, fsile, indent=4)

def sample_frames(flag, num_frames, dataset_name, train, config):

    # root_path = '/home/heyan/data2/Code/SSDG-CVPR2020-master/data_label/' + dataset_name
    # root_path = '/data2/heyan/Code/SSDG-CVPR2020-master/data_label/' + dataset_name
    root_path = config.data_path + dataset_name
    if (flag == 0):
        label_path = root_path + '/fake_label.json'
        save_label_path = root_path + '/choose_fake_label.json'
    elif (flag == 1):
        label_path = root_path + '/real_label.json'
        save_label_path = root_path + '/choose_real_label.json'
    elif (flag == 2):
        label_path = root_path + '/all_label.json'
        save_label_path = root_path + '/choose_all_label.json'
    elif (flag == 3):
        label_path = root_path + '/train_label.json'
        save_label_path = root_path + '/choose_train_label.json'
    elif (flag == 3_0):
        label_path = root_path + '/train_fake_label.json'
        save_label_path = root_path + '/choose_train_fake_label.json'
    elif (flag == 3_1):
        label_path = root_path + '/train_real_label.json'
        save_label_path = root_path + '/choose_train_real_label.json'
    elif (flag == 4):
        label_path = root_path + '/test_label.json'
        save_label_path = root_path + '/choose_test_label.json'
    elif (flag == 5):
        label_path = root_path + '/T_pseudo_label.json'
        save_label_path = root_path + '/choose_T_pseudo_label.json'
    elif (flag == 5_1):
        label_path = root_path + '/T_pseudo_real_label.json'
        save_label_path = root_path + '/choose_T_pseudo_real_label.json'
    elif (flag == 5_0):
        label_path = root_path + '/T_pseudo_fake_label.json'
        save_label_path = root_path + '/choose_T_pseudo_fake_label.json'
    elif (flag == 6):
        label_path = root_path + '/S_pseudo_label.json'
        save_label_path = root_path + '/choose_S_pseudo_label.json'
    elif (flag == 6_1):
        label_path = root_path + '/S_pseudo_real_label.json'
        save_label_path = root_path + '/choose_S_pseudo_real_label.json'
    elif (flag == 6_0):
        label_path = root_path + '/S_pseudo_fake_label.json'
        save_label_path = root_path + '/choose_S_pseudo_fake_label.json'

    all_label_json = json.load(open(label_path, 'r'))
    f_sample = open(save_label_path, 'w')  # 若没有，则创建该文件
    length = len(all_label_json)
    # three componets: frame_prefix, frame_num, png
    saved_frame_prefix = '/'.join(all_label_json[0]['photo_path'].split('/')[:-1])
    final_json = []
    video_number = 0
    single_video_frame_list = []
    single_video_label = 0
    for i in range(length):
        photo_path = all_label_json[i]['photo_path']
        photo_bbox = all_label_json[i]['photo_bbox']
        photo_label = all_label_json[i]['photo_label']
        photo_pseudo_label = all_label_json[i]['photo_pseudo_label']
        photo_confidence = all_label_json[i]['photo_confidence']
        photo_videoID = all_label_json[i]['photo_belong_to_video_ID']
        frame_prefix = '/'.join(photo_path.split('/')[:-1])

        # the last frame
        if (i == length - 1):
            photo_frame = photo_path.split('/')[-1]
            single_video_frame_list.append(photo_frame)
            single_video_label = photo_label
            single_video_pseudo_label = photo_pseudo_label
            single_video_confidence = photo_confidence
            single_video_videoID = photo_videoID

        # 第一步：读取同一视频的所有帧；
        # 第二步:当读取完毕后，选取相应帧
        if (frame_prefix != saved_frame_prefix or i == length - 1):
            # [1, 2, 3, 4,.....]
            single_video_frame_list.sort()
            single_video_frame_num = len(single_video_frame_list)
            num_frames = min(num_frames, single_video_frame_num)
            frame_interval = math.floor(single_video_frame_num / num_frames)
            image_id = random.randrange(0, frame_interval)
            if train == 'False':
                image_id = 0
            for j in range(num_frames):
                dict = {}
                dict['photo_path'] = saved_frame_prefix + '/' + single_video_frame_list[image_id + j * frame_interval]
                dict['photo_bbox'] = dict['photo_path'].split('.')[0] + "_bbox_mtccnn.txt"
                dict['photo_label'] = single_video_label
                dict['photo_pseudo_label'] = single_video_pseudo_label
                dict['photo_confidence'] = single_video_confidence
                dict['photo_belong_to_video_ID'] = single_video_videoID
                final_json.append(dict)
            video_number += 1
            saved_frame_prefix = frame_prefix
            single_video_frame_list.clear()
        # get every frame information
        photo_frame = photo_path.split('/')[-1]
        single_video_frame_list.append(photo_frame)
        single_video_label = photo_label
        single_video_pseudo_label = photo_pseudo_label
        single_video_confidence = photo_confidence
        single_video_videoID = photo_videoID

    if (flag == 0):
        print("Total video number(fake): ", video_number, dataset_name)
    elif (flag == 1):
        print("Total video number(real): ", video_number, dataset_name)
    else:
        print("Total video number(target): ", video_number, dataset_name)
    json.dump(final_json, f_sample, indent=4)
    f_sample.close()

    f_json = open(save_label_path)
    sample_data_pd = pd.read_json(f_json, dtype=False)
    return sample_data_pd

def sample_frames_depth(flag, num_frames, dataset_name, train):

    root_path = '/home/heyan/data2/Code/SSDG-CVPR2020-master/data_label/' + dataset_name
    dep_path = '/home/heyan/data2/Depth/'
    if (flag == 0):
        label_path = root_path + '/fake_label.json'
        save_label_path = root_path + '/choose_fake_label.json'
    elif (flag == 1):
        label_path = root_path + '/real_label.json'
        save_label_path = root_path + '/choose_real_label.json'
    elif (flag == 2):
        label_path = root_path + '/all_label.json'
        save_label_path = root_path + '/choose_all_label.json'
    elif (flag == 3):
        label_path = root_path + '/train_label.json'
        save_label_path = root_path + '/choose_train_label.json'
    elif (flag == 4):
        label_path = root_path + '/test_label.json'
        save_label_path = root_path + '/choose_test_label.json'
    elif (flag == 3_0):
        label_path = root_path + '/train_fake_label.json'
        save_label_path = root_path + '/choose_train_fake_label.json'
    elif (flag == 3_1):
        label_path = root_path + '/train_real_label.json'
        save_label_path = root_path + '/choose_train_real_label.json'


    all_label_json = json.load(open(label_path, 'r'))
    f_sample = open(save_label_path, 'w')  #若没有，则创建该文件
    length = len(all_label_json)
    # three componets: frame_prefix, frame_num, png
    saved_frame_prefix = '/'.join(all_label_json[0]['photo_path'].split('/')[:-1])
    final_json = []
    video_number = 0
    single_video_frame_list = []
    single_video_frame_num = 0
    single_video_label = 0
    for i in range(length):
        photo_path = all_label_json[i]['photo_path']
        photo_label = all_label_json[i]['photo_label']
        frame_prefix = '/'.join(photo_path.split('/')[:-1])
        # the last frame
        if (i == length - 1):
            photo_frame = int(photo_path.split('/')[-1].split('.')[0])
            single_video_frame_list.append(photo_frame)
            single_video_frame_num += 1
            single_video_label = photo_label
        # a new video, so process the saved one
        if (frame_prefix != saved_frame_prefix or i == length - 1):
            # [1, 2, 3, 4,.....]
            single_video_frame_list.sort()
            frame_interval = math.floor(single_video_frame_num / num_frames)
            if train == "True":
                num_frames = min(num_frames, single_video_frame_num)
                image_id = random.sample(range(0, single_video_frame_num), num_frames)
            else:
                image_id = [0]
            for j in range(num_frames):
                dict = {}
                # dict['photo_path'] = saved_frame_prefix + '/' + str(single_video_frame_list[j * frame_interval]).zfill(3) + '.png'  #选第6帧frame
                dict['photo_path'] = saved_frame_prefix + '/' + str(single_video_frame_list[image_id[j]]).zfill(3) + '.png'
                dict['photo_label'] = single_video_label
                dict['photo_belong_to_video_ID'] = video_number   #第几个视频
                # if single_video_label == 1:
                # depth_list = glob.glob(dep_path + dict['photo_path'].split('Datasets/')[1].split('/', 1)[1].rsplit('/', 1)[0] + '**/*.jpg', recursive=True)
                # dict['depth_path'] = depth_list[0]
                dict['depth_path'] = dep_path + dict['photo_path'].split('Datasets/')[1].split('/', 1)[1].rsplit('.', 1)[0] + '_depth.jpg'
                # else:
                #     dict['depth_path'] = '/home/heyan/data/Datasets/Depth/fake_depth/fake_depth.jpg'
                if not os.path.exists(dict['depth_path']):
                    print(dict['photo_path'])
                final_json.append(dict)
            video_number += 1
            saved_frame_prefix = frame_prefix
            single_video_frame_list.clear()
            single_video_frame_num = 0
        # get every frame information
        photo_frame = int(photo_path.split('/')[-1].split('.')[0])
        single_video_frame_list.append(photo_frame)
        single_video_frame_num += 1
        single_video_label = photo_label
    if(flag == 0):
        print("Total video number(fake): ", video_number, dataset_name)
    elif(flag == 1):
        print("Total video number(real): ", video_number, dataset_name)
    else:
        print("Total video number(target): ", video_number, dataset_name)
    json.dump(final_json, f_sample, indent=4)
    f_sample.close()

    f_json = open(save_label_path)
    sample_data_pd = pd.read_json(f_json)
    return sample_data_pd

def sample_frames_cross(flag, num_frames, dataset_name):

    root_path = '/home/heyan/data2/Code/SSDG-CVPR2020-master/data_label/' + dataset_name
    if(flag == 0): # select the train images
        label_path = root_path + '/train_label.json'
        save_label_path = root_path + '/choose_train_label.json'
    elif(flag == 1): # select the test images
        label_path = root_path + '/test_label.json'
        save_label_path = root_path + '/choose_test_label.json'

    all_label_json = json.load(open(label_path, 'r'))  #所有的fake/real/all/图像帧，根据label确定
    f_sample = open(save_label_path, 'w')  #若没有，则创建该文件
    length = len(all_label_json)
    # three componets: frame_prefix, frame_num, png
    saved_frame_prefix = '/'.join(all_label_json[0]['photo_path'].split('/')[:-1])  #图像帧前的路径
    final_json = []
    video_number = 0
    single_video_frame_list = []
    single_video_frame_num = 0
    single_video_label = 0
    for i in range(length):
        photo_path = all_label_json[i]['photo_path']
        photo_label = all_label_json[i]['photo_label']
        frame_prefix = '/'.join(photo_path.split('/')[:-1])  #图像帧前的路径
        # the last frame
        if (i == length - 1):  #最后一张
            photo_frame = int(photo_path.split('/')[-1].split('.')[0])
            single_video_frame_list.append(photo_frame)
            single_video_frame_num += 1
            single_video_label = photo_label
        # a new video, so process the saved one
        if (frame_prefix != saved_frame_prefix or i == length - 1):
            # [1, 2, 3, 4,.....]
            single_video_frame_list.sort()
            frame_interval = math.floor(single_video_frame_num / num_frames)
            image_id = np.random.randint(1, frame_interval)
            for j in range(num_frames):
                dict = {}
                dict['photo_path'] = saved_frame_prefix + '/' + str(single_video_frame_list[image_id]).zfill(3) + '.png'
                dict['photo_label'] = single_video_label
                dict['photo_belong_to_video_ID'] = video_number
                final_json.append(dict)
            video_number += 1
            saved_frame_prefix = frame_prefix
            single_video_frame_list.clear()
            single_video_frame_num = 0
        # get every frame information
        photo_frame = int(photo_path.split('/')[-1].split('.')[0])
        single_video_frame_list.append(photo_frame)
        single_video_frame_num += 1
        single_video_label = photo_label
    if(flag == 0):
        print("Total video number(train): ", video_number, dataset_name)  #每段视频选取一帧图像，即视频数等于最终图像帧数
    elif(flag == 1):
        print("Total video number(test): ", video_number, dataset_name)
    else:
        print("Total video number(target): ", video_number, dataset_name)
    json.dump(final_json, f_sample, indent=4)
    f_sample.close()

    f_json = open(save_label_path)
    sample_data_pd = pd.read_json(f_json)
    return sample_data_pd

def sample_frames_test(flag, num_frames, dataset_name):

    root_path = '/home/heyan/data2/Code/SSDG-CVPR2020-master/data_label/' + dataset_name
    if(flag == 0): # select the all images
        label_path = root_path + '/all_label.json'
        save_label_path = root_path + '/choose_all_label.json'
    elif(flag == 1): # select the test images
        label_path = root_path + '/test_label.json'
        save_label_path = root_path + '/choose_test_label.json'

    all_label_json = json.load(open(label_path, 'r'))  #所有的fake/real/all/图像帧，根据label确定
    f_sample = open(save_label_path, 'w')  #若没有，则创建该文件
    length = len(all_label_json)
    # three componets: frame_prefix, frame_num, png
    saved_frame_prefix = '/'.join(all_label_json[0]['photo_path'].split('/')[:-1])  #图像帧前的路径
    final_json = []
    video_number = 0
    single_video_frame_list = []
    single_video_frame_num = 0
    single_video_label = 0
    for i in range(length):
        photo_path = all_label_json[i]['photo_path']
        photo_label = all_label_json[i]['photo_label']
        frame_prefix = '/'.join(photo_path.split('/')[:-1])  #图像帧前的路径
        # the last frame
        if (i == length - 1):  #最后一张
            photo_frame = int(photo_path.split('/')[-1].split('.')[0])
            single_video_frame_list.append(photo_frame)
            single_video_frame_num += 1
            single_video_label = photo_label
        # a new video, so process the saved one
        if (frame_prefix != saved_frame_prefix or i == length - 1):
            # [1, 2, 3, 4,.....]
            single_video_frame_list.sort()
            for j in range(num_frames):
                dict = {}
                num = str(5)
                dict['photo_path'] = saved_frame_prefix + '/' + str(num.zfill(3)) + '.png' #选取第5帧
                dict['photo_label'] = single_video_label
                dict['photo_belong_to_video_ID'] = video_number
                final_json.append(dict)
            video_number += 1
            saved_frame_prefix = frame_prefix
            single_video_frame_list.clear()
            single_video_frame_num = 0
        # get every frame information
        photo_frame = int(photo_path.split('/')[-1].split('.')[0])
        single_video_frame_list.append(photo_frame)
        single_video_frame_num += 1
        single_video_label = photo_label
    if(flag == 0):
        print("Total video number(train): ", video_number, dataset_name)  #每段视频选取一帧图像，即视频数等于最终图像帧数
    elif(flag == 1):
        print("Total video number(test): ", video_number, dataset_name)
    else:
        print("Total video number(target): ", video_number, dataset_name)
    json.dump(final_json, f_sample, indent=4)
    f_sample.close()

    f_json = open(save_label_path)
    sample_data_pd = pd.read_json(f_json)
    return sample_data_pd

def sample_frames_intra(flag, num_frames, dataset_name):

    root_path = '/home/heyan/data2/Code/SSDG-CVPR2020-master/data_label/' + dataset_name
    if(flag == 0): # select the train fake images
        label_path = root_path + '/train_fake_label.json'
        save_label_path = root_path + '/choose_train_fake_label.json'
    elif(flag == 1): # select the train real images
        label_path = root_path + '/train_real_label.json'
        save_label_path = root_path + '/choose_train_real_label.json'

    all_label_json = json.load(open(label_path, 'r'))  #所有的fake/real/all/图像帧，根据label确定
    f_sample = open(save_label_path, 'w')  #若没有，则创建该文件
    length = len(all_label_json)
    # three componets: frame_prefix, frame_num, png
    saved_frame_prefix = '/'.join(all_label_json[0]['photo_path'].split('/')[:-1])  #图像帧前的路径
    final_json = []
    video_number = 0
    single_video_frame_list = []
    single_video_frame_num = 0
    single_video_label = 0
    for i in range(length):
        photo_path = all_label_json[i]['photo_path']
        photo_label = all_label_json[i]['photo_label']
        frame_prefix = '/'.join(photo_path.split('/')[:-1])  #图像帧前的路径
        # the last frame
        if (i == length - 1):  #最后一张
            photo_frame = int(photo_path.split('/')[-1].split('.')[0])
            single_video_frame_list.append(photo_frame)
            single_video_frame_num += 1
            single_video_label = photo_label
        # a new video, so process the saved one
        if (frame_prefix != saved_frame_prefix or i == length - 1):
            # [1, 2, 3, 4,.....]
            single_video_frame_list.sort()
            frame_interval = math.floor(single_video_frame_num / num_frames)
            image_id = np.random.randint(1, frame_interval)
            for j in range(num_frames):
                dict = {}
                dict['photo_path'] = saved_frame_prefix + '/' + str(single_video_frame_list[image_id]).zfill(3) + '.png'
                dict['photo_label'] = single_video_label
                dict['photo_belong_to_video_ID'] = video_number
                final_json.append(dict)
            video_number += 1
            saved_frame_prefix = frame_prefix
            single_video_frame_list.clear()
            single_video_frame_num = 0
        # get every frame information
        photo_frame = int(photo_path.split('/')[-1].split('.')[0])
        single_video_frame_list.append(photo_frame)
        single_video_frame_num += 1
        single_video_label = photo_label
    if(flag == 0):
        print("Total video number(train_fake): ", video_number, dataset_name)  #每段视频选取一帧图像，即视频数等于最终图像帧数
    elif(flag == 1):
        print("Total video number(train_real): ", video_number, dataset_name)
    else:
        print("Total video number(test): ", video_number, dataset_name)
    json.dump(final_json, f_sample, indent=4)
    f_sample.close()

    f_json = open(save_label_path)
    sample_data_pd = pd.read_json(f_json)
    return sample_data_pd

def sample_frames_devep(num_frames, dataset_name):
    save_label_path = '/home/heyan/data2/Code/SSDG-CVPR2020-master/data_label/' + ''.join(dataset_name) + '_choose_devep_label.json'
    f_sample = open(save_label_path, 'w')  # 若没有，则创建该文件
    final_json = []

    for iter in range(0, len(dataset_name)):
        root_path = '/home/heyan/data2/Code/SSDG-CVPR2020-master/data_label/' + dataset_name[iter]
        if dataset_name[iter] == 'replay' or dataset_name[iter] == 'oulu':
            label_path_devep = root_path + '/valid_label.json'
        else:
            label_path_devep = root_path + '/test_label.json'

        all_label_json = json.load(open(label_path_devep, 'r'))  # 所有的图像帧
        length = len(all_label_json)
        # three componets: frame_prefix, frame_num, png
        saved_frame_prefix = '/'.join(all_label_json[0]['photo_path'].split('/')[:-1])  # 图像帧前的路径

        video_number = 0
        single_video_frame_list = []
        single_video_frame_num = 0
        single_video_label = 0
        for i in range(length):
            photo_path = all_label_json[i]['photo_path']
            photo_label = all_label_json[i]['photo_label']
            frame_prefix = '/'.join(photo_path.split('/')[:-1])  # 图像帧前的路径
            # the last frame
            if (i == length - 1):  # 最后一张
                photo_frame = int(photo_path.split('/')[-1].split('.')[0])
                single_video_frame_list.append(photo_frame)
                single_video_frame_num += 1
                single_video_label = photo_label
            # a new video, so process the saved one
            if (frame_prefix != saved_frame_prefix or i == length - 1):
                # [1, 2, 3, 4,.....]
                single_video_frame_list.sort()
                frame_interval = math.floor(single_video_frame_num / num_frames)
                for j in range(num_frames):
                    dict = {}
                    # dict['photo_path'] = saved_frame_prefix + '/' + str(single_video_frame_list[ j * frame_interval]) + '.png'
                    num = str(5)
                    dict['photo_path'] = saved_frame_prefix + '/' + str(num.zfill(3)) + '.png'  # 选取第5帧
                    dict['photo_label'] = single_video_label
                    dict['photo_belong_to_video_ID'] = video_number
                    final_json.append(dict)
                video_number += 1
                saved_frame_prefix = frame_prefix
                single_video_frame_list.clear()
                single_video_frame_num = 0
            # get every frame information
            photo_frame = int(photo_path.split('/')[-1].split('.')[0])
            single_video_frame_list.append(photo_frame)
            single_video_frame_num += 1
            single_video_label = photo_label
        print("Total video number(valid): ", video_number, dataset_name[iter])  # 每段视频选取一帧图像，即视频数等于最终图像帧数

    json.dump(final_json, f_sample, indent=4)
    f_sample.close()

    f_json = open(save_label_path)
    sample_data_pd = pd.read_json(f_json)
    return sample_data_pd


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def mkdirs(checkpoint_path, best_model_path, logs):
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    if not os.path.exists(best_model_path):
        os.makedirs(best_model_path)
    if not os.path.exists(logs):
        os.mkdir(logs)

def time_to_str(t, mode='min'):
    if mode=='min':
        t  = int(t)/60
        hr = t//60
        min = t%60
        return '%2d hr %02d min'%(hr,min)
    elif mode=='sec':
        t   = int(t)
        min = t//60
        sec = t%60
        return '%2d min %02d sec'%(min,sec)
    else:
        raise NotImplementedError

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.file = None

    def open(self, file, mode=None):
        if mode is None:
            mode = 'w'
        self.file = open(file, mode)
    def write(self, message, is_terminal=1, is_file=1):
        if '\r' in message:
            is_file = 0
        if is_terminal == 1:
            self.terminal.write(message)
            self.terminal.flush()
        if is_file == 1:
            self.file.write(message)
            self.file.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass

def save_checkpoint(save_list, is_best, model, gpus, checkpoint_path, best_model_path, filename='_checkpoint.pth.tar'):
    epoch = save_list[0]
    valid_args = save_list[1]
    best_model_HTER = round(save_list[2], 5)
    best_model_ACC = save_list[3]
    best_model_ACER = save_list[4]
    threshold = save_list[5]
    if(len(gpus) > 1):
        old_state_dict = model.state_dict()
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in old_state_dict.items():
            flag = k.find('.module.')
            if (flag != -1):
                k = k.replace('.module.', '.')
            new_state_dict[k] = v
        state = {
            "epoch": epoch,
            "state_dict": new_state_dict,
            "valid_arg": valid_args,
            "best_model_EER": best_model_HTER,
            "best_model_ACER": best_model_ACER,
            "best_model_ACC": best_model_ACC,
            "threshold": threshold
        }
    else:
        state = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "valid_arg": valid_args,
            "best_model_EER": best_model_HTER,
            "best_model_ACER": best_model_ACER,
            "best_model_ACC": best_model_ACC,
            "threshold": threshold
        }
    filepath = checkpoint_path + filename
    torch.save(state, filepath)
    # just save best model
    if is_best:
        shutil.copy(filepath, best_model_path + 'model_best_' + str(best_model_HTER) + '_' + str(epoch) + '.pth.tar')

def zero_param_grad(params):
    for p in params:
        if p.grad is not None:
            p.grad.zero_()