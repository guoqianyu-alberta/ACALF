"""Dataset for evaluation"""

from PIL import Image
from torch.utils.data import Dataset
import os
import numpy as np
import torch
import torch.optim
import torch.nn.functional as F


class DatasetCustom(Dataset):
    def __init__(self, base_dir, n_shots, transform, max_iters, dataset_name, class_names, use_original_imgsize):
        super().__init__()
        self.base_dir = base_dir
        self.ids = []
        self.shots = n_shots
        self.queries = 1
        self.max_iters = max_iters
        self.cls_names = class_names
        self.dataset_name = dataset_name
        self.benchmark = dataset_name
        self.transform = transform
        self.class_ids = [id for id in range(len(self.cls_names))]
        self.use_original_imgsize = use_original_imgsize
        # print(self.class_ids)

        temp_ids = []
        cls_ids = []

        if dataset_name == 'Animal' or dataset_name == 'Magnetic_tile_surface' or dataset_name == 'Artificial_Luna_Landscape' or dataset_name == 'Aerial':
            self.path_dir = os.path.join(self.base_dir, dataset_name)
        else:
            self.path_dir = self.base_dir

        for name in self.cls_names:
            for id in os.listdir(os.path.join(self.path_dir, name,'support','images')):
                temp_ids.append(id)
            cls_ids.append(temp_ids)
            temp_ids = []
            for id in os.listdir(os.path.join(self.path_dir, name,'query','images')):
                temp_ids.append(id)
            cls_ids.append(temp_ids)
            self.ids.append(cls_ids)
            cls_ids = []
            temp_ids = []

    def __len__(self):
        return self.max_iters

    def __getitem__(self, idx):
        support_names, query_name, class_sample = self.sample_episode(idx)

        query_mask_name = query_name.split('.')[0] + ".png"
        query_img = Image.open(os.path.join(self.path_dir, self.cls_names[class_sample], 'query', 'images', query_name))
        org_qry_imsize = query_img.size
        query_img = self.transform(self.process_image(query_img))
        # print(np.unique(query_img))
        query_mask = Image.open(os.path.join(self.path_dir, self.cls_names[class_sample], 'query', 'masks', query_mask_name))
        query_mask = self.process_mask(query_mask, class_sample)
        if not self.use_original_imgsize:
            # (384, 384)
            query_mask = F.interpolate(query_mask.unsqueeze(0).unsqueeze(0).float(), query_img.size()[-2:], mode='nearest').squeeze()
        query_ignore_idx = self.extract_ignore_idx(query_mask)

        support_mask_names = [name.split('.')[0] + ".png" for name in support_names]
        support_imgs = [Image.open(os.path.join(self.path_dir, self.cls_names[class_sample], 'support', 'images', name)) for name in support_names]
        support_imgs = torch.stack([self.transform(self.process_image(support_img)) for support_img in support_imgs])

        support_cmasks = [Image.open(os.path.join(self.path_dir, self.cls_names[class_sample], 'support', 'masks', name)) for name in
                        support_mask_names]
        support_cmasks = [self.process_mask(mask, class_sample) for mask in support_cmasks]
        support_ignore_idxs = []
        support_masks = []
        for scmask in support_cmasks:
            scmask = F.interpolate(scmask.unsqueeze(0).unsqueeze(0).float(), support_imgs.size()[-2:],
                                   mode='nearest').squeeze()
            support_masks.append(scmask)
            support_ignore_idx = self.extract_ignore_idx(scmask)
            support_ignore_idxs.append(support_ignore_idx)
        support_masks = torch.stack(support_masks)
        support_ignore_idxs = torch.stack(support_ignore_idxs)

        batch = {'query_img': query_img,
                 'query_mask': query_mask,
                 'query_name': [query_name],
                 'query_ignore_idx': query_ignore_idx,

                 'org_query_imsize': org_qry_imsize,
                 'support_imgs': support_imgs,
                 'support_masks': support_masks,
                 'support_ignore_idxs': support_ignore_idxs,
                 'class_id': torch.tensor(class_sample)
                 }
        return batch

    def extract_ignore_idx(self, mask):
        boundary = (mask / 255).floor()
        return boundary

    def process_mask(self, label, class_id):
        if label.mode != 'L':
            label = label.convert('L')
        label = torch.tensor(np.array(label))
        label = torch.where(label == 255, torch.ones_like(label), torch.zeros_like(label))
        return label

    def process_image(self, image):
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image
    def sample_episode(self, idx):
        class_id = idx % len(self.cls_names)
        support_names = np.random.choice(self.ids[class_id][0], self.shots, replace=False)
        query_name = np.random.choice(self.ids[class_id][1], 1, replace=False)
        return support_names, query_name[0], class_id












