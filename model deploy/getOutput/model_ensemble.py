import os
import math
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import open_clip
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as T
from tqdm import tqdm
import pandas as pd
import cv2
from PIL import Image
from get_image import *


path1 = 'model/model_epoch_4_mAP3_0.34.pt'

def read_img(img_path, is_gray=False):
    mode = cv2.IMREAD_COLOR if not is_gray else cv2.IMREAD_GRAYSCALE
    img = cv2.imread(img_path, mode)
    if not is_gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

class ProductDataset(Dataset):
    def __init__(self,
                 img_dir,
                 annotations_file,
                 transform=None,
                 final_transform=None,
                 headers=None,
                 test_mode=False):
        self.data = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.final_transform = final_transform
        self.headers = {"img_path": "img_path", "product_id": "product_id"}
        if headers:
            self.headers = headers
        self.test_mode = test_mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.data[self.headers["img_path"]][idx])

        img = read_img(img_path)
        if self.test_mode:
            x, y, w, h = self.data["bbox_x"][idx], self.data["bbox_y"][idx], \
                         self.data["bbox_w"][idx], self.data["bbox_h"][idx]
            img = img[y:y+h, x:x+w]


        if self.transform is not None:
            img = cv2.transform(image=img)["image"]

        if self.final_transform is not None:
            if isinstance(img, np.ndarray):
                img =  Image.fromarray(img)
            img = self.final_transform(img)

        product_id = self.data[self.headers["product_id"]][idx]
        return img, product_id

def get_final_transform():
    final_transform = T.Compose([
            T.Resize(
                size=(224, 224),
                interpolation=T.InterpolationMode.BICUBIC,
                antialias=True),
            T.ToTensor(),
            T.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711)
            )
        ])
    return final_transform

@th.no_grad()
def extract_embeddings(model, dataloader, epoch=10, use_cpu=True):
    features = []
    product_id = []

    for _ in range(epoch):
        for imgs, p_id in tqdm(dataloader):
            if use_cpu:
                imgs = imgs.cuda()
            features.append(th.squeeze(model(imgs.half())).detach().cpu().numpy().astype(np.float32))
            product_id.append(th.squeeze(p_id).detach().cpu().numpy())


    return np.concatenate(features, axis=0), np.concatenate(product_id)









