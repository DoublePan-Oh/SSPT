import argparse
import copy
import json
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import torch.utils.data
from util.misc import nested_tensor_from_tensor_list

def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data

def augmentation(feature, out_size, offset):
    if feature.size(1) <= out_size:
            upsampled_size = feature.size(1) * 4
            initial_size = feature.size(1)
            # initial_size = 10
            tile_idx = np.zeros(initial_size)
            for i in range(initial_size):
                for j in range(math.floor(upsampled_size/initial_size)):
                    tile_idx[i] += 1
            for i in range(initial_size):
                tile_idx[i] += 1
                if np.sum(tile_idx) == upsampled_size: break
            feature = feature.repeat_interleave(torch.LongTensor(tile_idx),dim=1)
            # print(feature.shape,"$$$$$$$$$$$")
            fidx = sorted(random.sample(list(range(0,feature.size(1))),initial_size))
            feature = feature[:,torch.tensor(fidx)].squeeze(0)
            # print(feature.shape,"~~~~~~~~~~~~~")
        
    elif feature.size(1) > out_size: 
        if len(list(range(0+offset,feature.size(1)-offset))) < out_size: 
            # exit()
            initial_size = len(list(range(0+offset,feature.size(1)-offset)))
            tile_idx = np.zeros(initial_size)
            for i in range(initial_size):
                for j in range(math.floor(out_size/feature.size(1))):
                    tile_idx[i] += 1

            for i in range(initial_size):
                tile_idx[i] += 1
                if np.sum(tile_idx) == out_size: break

            sample_idx = torch.tensor(list(range(0+offset,feature.size(1)-offset))).repeat_interleave(torch.LongTensor(tile_idx))
            fidx = sorted(sample_idx.tolist())
        else:
            fidx = sorted(random.sample(list(range(0+offset,feature.size(1)-offset)),out_size))

        feature_sampled = []
        for idx in fidx:
            feature_sampled.append(torch.mean(feature[:,torch.tensor(list(range(idx-offset,idx+offset)))],dim=1).unsqueeze(1))
        feature = torch.cat(feature_sampled,dim=1)
    return feature

class VideoRecord:
    def __init__(self, vid, num_frames, locations, s_e_scores, gt, fps, args, patches):
        self.id = vid
        self.locations = locations
        self.base = float(locations[0])
        # print(self.base)
        self.window_size = args.window_size
        self.interval = args.interval
        self.locations_norm = [
            (i - self.base) / (self.window_size * self.interval)
            for i in locations
        ]
        self.locations_offset = [
            location - self.base for location in locations
        ]
        self.num_frames = num_frames
        self.absolute_position = args.absolute_position

        self.gt = gt
        self.gt_norm = copy.deepcopy(gt)
        self.patches = patches
        # normalize gt start and end
        for i in self.gt_norm:
            i[0][0] = (i[0][0] - self.base) / (self.window_size *
                                               self.interval)
            i[0][1] = (i[0][1] - self.base) / (self.window_size *
                                               self.interval)
            # print(i[0][0],"  ",i[0][1])
            # i[0][0] = (i[0][0]) / (num_frames)
            # i[0][1] = (i[0][1]) / (num_frames)
            # print(i[0][0],"  ",i[0][1])
        # exit()
        self.gt_s_e_frames = [i[0] for i in self.gt_norm]
        # print("self.gt_s_e_frames:",self.gt_s_e_frames)

        self.fps = fps
        self.duration = num_frames / fps

        if (args.point_prob_normalize is True):
            range_start = np.max(s_e_scores[:, 0]) - np.min(s_e_scores[:, 0])
            range_end = np.max(s_e_scores[:, 1]) - np.min(s_e_scores[:, 1])
            s_e_scores[:, 0] = (s_e_scores[:, 0] -
                                np.min(s_e_scores[:, 0])) / range_start
            s_e_scores[:, 1] = (s_e_scores[:, 1] -
                                np.min(s_e_scores[:, 1])) / range_end
        self.s_e_scores = s_e_scores

import random
import math
from scipy.interpolate import interp1d
def resizeFeature(inputData,newSize):
    # inputX: (temporal_length,feature_dimension) #
    # print(inputData.shape)
    # inputData = inputData.squeeze_(0).squeeze_(0).permute(1,0)
    originalSize=len(inputData)
    # print(originalSize)
    if originalSize==1:
        # print("shape:",inputData.shape)
        inputData=np.reshape(inputData,[-1])
        # print("inputData.shape:",inputData.shape)
        return np.stack([inputData]*newSize)
    x=np.array(range(originalSize))
    f=interp1d(x,inputData,axis=0)
    x_new=[i*float(originalSize-1)/(newSize-1) for i in range(newSize)]
    y_new=f(x_new)
    # y_new = torch.tensor(y_new,dtype=torch.float).squeeze_(1).squeeze_(1).permute(1,0)
    return y_new




class ThumosDetection(torch.utils.data.Dataset):
    def __init__(self, feature_folder, tem_folder, anno_file, split, args):
        annotations = load_json(anno_file)
        video_list = annotations.keys()
        self.window_size = args.window_size
        self.feature_folder = feature_folder
        self.tem_folder = tem_folder
        self.anno_file = load_json(anno_file)
        self.num_gt = args.gt_size
        self.num_patches = args.num_patches
        if split == 'val':
            self.split = 'test'
        else:
            self.split = 'val'
        self.video_dict = {}
        video_pool = list(self.anno_file.keys())
        video_pool.sort()
        self.video_dict = {video_pool[i]: i for i in range(len(video_pool))}

        

        self.video_list = []
        offset = 32
        outsize = 256
        for vid in video_list:
            if self.split in vid:
                num_frames = int(self.anno_file[vid]['duration_frame'])
                fps = int(self.anno_file[vid]['fps'])
                vid_feature = torch.load(os.path.join(self.feature_folder, vid))

                gt_annotations = [
                    item['segment_frame']
                    for item in self.anno_file[vid]['annotations']
                ]
                gt_labels = [
                    int(item['label'])
                    for item in self.anno_file[vid]['annotations']
                ]
                length = []
                annotations =[]
                labels = []
                patches = []
                if len(gt_annotations) == 1:
                    continue
                # print("len(gt_annotations):",len(gt_annotations))

                # while len(length) < self.num_patches: # 10 is num_patch
                for idx in range(len(gt_annotations)):

                    start = gt_annotations[idx][0]
                    end = gt_annotations[idx][1]
                    labels.append(gt_labels[idx])
                    pesudo_num_length = end - start
                    T_start =  int(start // 8 - 7)
                    T_end =  int(end // 8 - 7 )
                    if end > num_frames or T_start < 0 or T_end > vid_feature.shape[0] or T_start == T_end:
                        continue
                    segment_frame = [start, end]
                    # print(num_frames)
                    # print(segment_frame)
                    annotations.append(segment_frame)
                    # labels.append(1) 
                    
                    length.append(pesudo_num_length)
                    indice = torch.tensor([i for i in range(T_start , T_end )])
                    patch = torch.index_select(vid_feature, 0, indice)
                    patch = patch.permute(1,0)
                    patch = augmentation(patch, outsize, offset)
                    patch = patch.permute(1,0)
                    patch = torch.tensor(resizeFeature(patch,20),dtype=torch.float32).squeeze_(1).squeeze_(1)
                    # print("@@@@patch.shape:",patch.shape) # torch.Size([20, 2048])
                    
                    patches.append(patch)
                    # print(pesudo_num_length)
                    # print(indice.shape)
                    # indices.append(indice)
                if len(patches) == 1:
                    continue
                s_e_seq = pd.read_csv(
                    os.path.join(self.tem_folder, vid + '.csv'))
                start_scores = np.expand_dims(s_e_seq.start.values, 1)
                end_scores = np.expand_dims(s_e_seq.end.values, 1)
                frames = np.expand_dims(s_e_seq.frame.values, 1)
                seq_len = len(s_e_seq)
                if seq_len <= self.window_size:
                    locations = np.zeros((self.window_size, 1))
                    locations[:seq_len, :] = frames
                    s_e_scores = np.zeros((self.window_size, 2))
                    s_e_scores[:seq_len, 0] = start_scores.squeeze()
                    s_e_scores[:seq_len, 1] = end_scores.squeeze()
                    gt = [(annotations[idx], labels[idx])
                            for idx in range(len(annotations))]
                    self.video_list.append(
                        VideoRecord(vid, num_frames, locations, s_e_scores, gt,
                                    fps, args, patches))
                
                # for idx in range(len(annotations)):
                #     gt = []
                #     anno = annotations[idx]
                #     label = labels[idx]
                #     gt.append((anno, label))
                #     self.video_list.append(VideoRecord(vid, num_frames, gt,
                #             fps, args, patches))

                else:
                    if self.split == 'test':
                        overlap_ratio = 2
                    else:
                        overlap_ratio = 4
                    stride = self.window_size // overlap_ratio
                    ws_starts = [
                        i * stride
                        for i in range((seq_len // self.window_size - 1) *
                                       overlap_ratio + 1)
                    ]
                    ws_starts.append(seq_len - self.window_size)

                    for ws in ws_starts:
                        locations = frames[ws:ws + self.window_size]
                        s_scores = start_scores[ws:ws + self.window_size]
                        e_scores = end_scores[ws:ws + self.window_size]
                        s_e_scores = np.concatenate((s_scores, e_scores),
                                                    axis=1)

                        gt = []
                        for idx in range(len(annotations)):
                            anno = annotations[idx]
                            label = labels[idx]
                            if anno[0] >= locations[0] and anno[
                                    1] <= locations[-1]:
                                # print("locations[0]:",locations[0])
                                # print("locations[-1]:",locations[-1])
                                gt.append((anno, label))
                        if self.split == 'test':
                            self.video_list.append(
                                VideoRecord(vid, num_frames, locations, s_e_scores, gt,
                                            fps, args, patches))
                        elif len(gt) > 0:
                            self.video_list.append(
                                VideoRecord(vid, num_frames, locations, s_e_scores,  gt,
                                             fps, args, patches))
        print(split, len(self.video_list))

    def get_data(self, video: VideoRecord):
        '''
        :param VideoRecord
        :return vid_name,
        locations : [N, 1],
        all_props_feature: [N, ft_dim + 2 + pos_dim],
        (gt_start_frame, gt_end_frame): [num_gt, 2]
        '''

        vid = video.id
        num_frames = video.num_frames
        base = video.base
        patches = video.patches

        og_locations = torch.Tensor([location for location in video.locations])
        vid_feature = torch.load(os.path.join(self.feature_folder, vid))
        # print("vid_feature.shape:",vid_feature.shape)

        ft_idxes = [
            min(i // 8, vid_feature.shape[0] - 1) for i in og_locations
        ]


        # for j in indices:
        #     print(j)
        #     # exit()
        #     patch = torch.index_select(vid_feature, 0, j)
        #     # patch = torch.tensor(resizeFeature(patch, 100),dtype=torch.float32).squeeze_(1).squeeze_(1).permute(1,0)
        #     # print(patch.shape)
        #     patches.append(patch)
        # # exit()
        
        patches = torch.stack(patches)
        # print("before patches.shape", patches.shape)
        patches =  torch.tensor(resizeFeature(patches,4),dtype=torch.float32)
        # print("after patches.shape", patches.shape)
        # exit()
        snippet_fts = []
        for i in ft_idxes:
            i = int(i)
            snippet_fts.append(vid_feature[i].squeeze())
        snippet_fts = torch.stack(snippet_fts)

        # fts = torch.tensor(resizeFeature(vid_feature,100),dtype=torch.float32).squeeze_(1).squeeze_(1)
        # print("fts.shape:",fts.shape)
        # snippet_fts.append(fts)
        # snippet_fts = torch.squeeze(torch.stack(snippet_fts))
        # print("snippet_fts.shape:",snippet_fts.shape)
        # exit()
        assert snippet_fts.shape == (self.window_size,
                                     2048), print(snippet_fts.shape)

        if video.absolute_position:
            locations = torch.Tensor(
                [location for location in video.locations])
        else:
            locations = torch.Tensor(
                [location for location in video.locations_offset])

        s_e_scores = torch.Tensor(video.s_e_scores)
        # s_e_scores = torch.ones([100,2])

        gt_s_e_frames = [(s, e, 0) for (s, e) in video.gt_s_e_frames]
        for (s, e, _) in gt_s_e_frames:
            assert s >= 0 and s <= 1 and e >= 0 and e <= 1, '{} {}'.format(
                s, e)

        targets = {
            'labels': [],
            'boxes': [],
            'video_id': torch.Tensor([self.video_dict[vid]])
        }
        for (start, end, label) in gt_s_e_frames:
            targets['labels'].append(int(label))
            targets['boxes'].append((start, end))

        targets['labels'] = torch.LongTensor(targets['labels'])

        targets['boxes'] = torch.Tensor(targets['boxes'])

        # all_props_feature = torch.cat((snippet_fts, s_e_scores), dim=1)

        # print("targets:",targets)
        return vid,  snippet_fts, locations, targets, num_frames, s_e_scores, patches, base

    def __getitem__(self, idx):
        # print(self.get_data(self.video_list[idx]))
        return self.get_data(self.video_list[idx])

    def __len__(self):
        return len(self.video_list)


def collate_fn(batch):
    vid_name_list, target_list, num_frames_list, base_list = [[]
                                                              for _ in range(4)
                                                              ]
    batch_size = len(batch)
    # print(batch)
    
    ft_dim = batch[0][1].shape[-1]
    # print(ft_dim)
    max_props_num = batch[0][1].shape[0]
    # print(max_props_num)

    # exit()
    # props_features = torch.zeros(batch_size, max_props_num, ft_dim)
    snippet_fts = torch.zeros(batch_size, max_props_num, ft_dim)
    locations = torch.zeros(batch_size, max_props_num, 1, dtype=torch.double)
    s_e_scores = torch.zeros(batch_size, max_props_num, 2)
    patches = torch.zeros(batch_size, 4, 20, ft_dim) # 8 is num_patch, 100 is (segment length //8)  (8 frame is a feature chunck)
    for i, sample in enumerate(batch):

        vid_name_list.append(sample[0])
        target_list.append(sample[3])
        snippet_fts[i, :max_props_num, :] = sample[1]
        patches[i, :20, :] = sample[6] #################################
        locations[i, :max_props_num, :] = sample[2].reshape((-1, 1))
        num_frames_list.append(sample[4])
        if (sample[5] is not None):
            base_list.append(sample[7])
       
        s_e_scores[i, :max_props_num, :] = sample[5]

    num_frames = torch.from_numpy(np.array(num_frames_list))
    base = torch.from_numpy(np.array(base_list))

    batch = list(zip(*batch))
    batch[1] = nested_tensor_from_tensor_list(batch[1])
    # print("ba.tensors.shape:", ba.tensors.shape)
    # print("ba.mask.shape:",ba.mask.shape)
    # batch[1] = torch.stack(batch[1], dim=0)
    return vid_name_list, batch[1], locations, target_list, num_frames,  s_e_scores, patches, base


def build(split, args):
    # split = train/val
    root = Path(args.feature_path)
    assert root.exists(
    ), f'provided thumos14 feature path {root} does not exist'
    feature_folder = root
    tem_folder = Path(args.tem_path)
    anno_file = Path(args.annotation_path)

    dataset = ThumosDetection(feature_folder, tem_folder, anno_file, split,
                              args)
    return dataset


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector',
                                     add_help=False)
    parser.add_argument('--batch_size', default=2, type=int)

    # dataset parameters
    parser.add_argument('--dataset_file', default='thumos14')
    parser.add_argument('--window_size', default=100, type=int)
    parser.add_argument('--gt_size', default=100, type=int)
    parser.add_argument('--feature_path',
                        default='/data1/tj/thumos_2048/',
                        type=str)
    parser.add_argument('--tem_path',
                        default='/data1/tj/BSN_share/output/TEM_results',
                        type=str)
    parser.add_argument('--annotation_path',
                        default='thumos14_anno_action.json',
                        type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--num_workers', default=2, type=int)

    return parser
