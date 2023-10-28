import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import os
import faiss
import copy
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image
from torchvision import transforms
import cv2
import time

class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=30.0,
                 m=0.50, easy_margin=False, ls_eps=0.0, device=torch.device('cuda')):
        super(ArcMarginProduct, self).__init__()
        self.device = device
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.ls_eps = ls_eps  # label smoothing
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device=self.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        # -------------torch.where(out_i = {x_i if condition_i else y_i) ------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output

class DenseCrossEntropy(nn.Module):
    def forward(self, x, target):
        x = x.float()
        target = target.float()
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        loss = -logprobs * target
        loss = loss.sum(-1)
        return loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, x, target):
        x = x.float()
        target = target.float()
        probs = torch.nn.functional.softmax(x, dim=-1)
        logprobs = torch.log(probs)

        loss = -logprobs * target * (1 - probs) ** self.gamma
        loss = loss.sum(-1)
        return loss.mean()

class ArcMarginProduct_subcenter(nn.Module):
    def __init__(self, in_features, out_features, k=3):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features*k, in_features))
        self.reset_parameters()
        self.k = k
        self.out_features = out_features

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, features):
        cosine_all = F.linear(F.normalize(features), F.normalize(self.weight))
        cosine_all = cosine_all.view(-1, self.out_features, self.k)
        cosine, _ = torch.max(cosine_all, dim=2)
        return cosine

class ArcFaceLossAdaptiveMargin(nn.modules.Module):
    def __init__(self, margins, s=30.0, crit='ce'):
        super().__init__()
        if crit == 'ce':
            self.crit = DenseCrossEntropy()
        else:
            self.crit = FocalLoss()
        self.s = s
        self.margins = margins

    def forward(self, logits, labels, out_dim):
        ms = []
        ms = self.margins[labels.cuda().numpy()]
        cos_m = torch.from_numpy(np.cos(ms)).float().cuda()
        sin_m = torch.from_numpy(np.sin(ms)).float().cuda()
        th = torch.from_numpy(np.cos(math.pi - ms)).float().cuda()
        mm = torch.from_numpy(np.sin(math.pi - ms) * ms).float().cuda()
        labels = F.one_hot(labels, out_dim).float()
        logits = logits.float()
        cosine = logits
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * cos_m.view(-1,1) - sine * sin_m.view(-1,1)
        phi = torch.where(cosine > th.view(-1,1), phi, cosine - mm.view(-1,1))
        output = (labels * phi) + ((1.0 - labels) * cosine)
        output *= self.s
        loss = self.crit(output, labels)
        return loss

def set_seed(seed):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_similiarity_hnsw(embeddings_gallery, emmbeddings_query, k):
    # this is guy is really fast
    print('Processing indices...')

    s = time.time()
    index = faiss.IndexHNSWFlat(embeddings_gallery.shape[1], 32)
    index.add(embeddings_gallery)

    scores, indices = index.search(emmbeddings_query, k)
    e = time.time()

    print(f'Finished processing indices, took {e - s}s')
    return scores, indices

def get_similiarity_l2(embeddings_gallery, emmbeddings_query, k):
    print('Processing indices...')

    s = time.time()
    index = faiss.IndexFlatL2(embeddings_gallery.shape[1])
    index.add(embeddings_gallery)

    scores, indices = index.search(emmbeddings_query, k)
    e = time.time()

    print(f'Finished processing indices, took {e - s}s')
    return scores, indices


def get_similiarity_IP(embeddings_gallery, emmbeddings_query, k):
    print('Processing indices...')

    s = time.time()
    index = faiss.IndexFlatIP(embeddings_gallery.shape[1])
    index.add(embeddings_gallery)

    scores, indices = index.search(emmbeddings_query, k)
    e = time.time()

    print(f'Finished processing indices, took {e - s}s')
    return scores, indices

def get_similiarity(embeddings, k):
    print('Processing indices...')

    index = faiss.IndexFlatL2(embeddings.shape[1])

    res = faiss.StandardGpuResources()

    index = faiss.index_cpu_to_gpu(res, 0, index)

    index.add(embeddings)

    scores, indices = index.search(embeddings, k)
    print('Finished processing indices')

    return scores, indices

def map_per_image(label, predictions, k=5):
    try:
        return 1 / (predictions[:k].index(label) + 1)
    except ValueError:
        return 0.0

def map_per_set(labels, predictions, k=5):
    return np.mean([map_per_image(l, p, k) for l,p in zip(labels, predictions)])

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, window_size=None):
        self.length = 0
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.window_size = window_size

    def reset(self):
        self.length = 0
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if self.window_size and (self.count >= self.window_size):
            self.reset()
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_lr_groups(param_groups):
        groups = sorted(set([param_g['lr'] for param_g in param_groups]))
        groups = ["{:2e}".format(group) for group in groups]
        return groups

def convert_indices_to_labels(indices, labels):
    indices_copy = copy.deepcopy(indices)
    for row in indices_copy:
        for j in range(len(row)):
            row[j] = labels[row[j]]
    return indices_copy

class Multisample_Dropout(nn.Module):
    def __init__(self, dropout_rate=0.1):
        super(Multisample_Dropout, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.dropouts = nn.ModuleList([nn.Dropout((i+1)*.1) for i in range(5)])

    def forward(self, x, module):
        x = self.dropout(x)
        return torch.mean(torch.stack([module(dropout(x)) for dropout in self.dropouts],dim=0),dim=0)

def transforms_auto_augment(image_path, image_size):
    image = Image.open(image_path).convert('RGB')
    train_transforms = transforms.Compose([transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET), transforms.PILToTensor()])
    return train_transforms(image)

def transforms_cutout(image_path, image_size):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.uint8)
    train_transforms = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ImageCompression(quality_lower=99, quality_upper=100),
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=10, border_mode=0, p=0.7),
            A.Resize(image_size, image_size),
            A.Cutout(max_h_size=int(image_size * 0.4), max_w_size=int(image_size * 0.4), num_holes=1, p=0.5),
            ToTensorV2(),
        ])
    return train_transforms(image=image)['image']

def transforms_happy_whale(image_path, image_size):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.uint8)
    aug8p3 = A.OneOf([
            A.Sharpen(p=0.3),
            A.ToGray(p=0.3),
            A.CLAHE(p=0.3),
        ], p=0.5)

    train_transforms = A.Compose([
            A.ShiftScaleRotate(rotate_limit=15, scale_limit=0.1, border_mode=cv2.BORDER_REFLECT, p=0.5),
            A.Resize(image_size, image_size),
            aug8p3,
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            ToTensorV2(),
        ])
    return train_transforms(image=image)['image']

def transforms_valid(image_path, image_size):
    image = Image.open(image_path).convert('RGB')
    valid_transforms = transforms.Compose([transforms.PILToTensor()])
    return valid_transforms(image)


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
            img = transform(image=img)["image"]

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
            features.append(th.squeeze(model(imgs.half())).detach().cuda().numpy().astype(np.float32))
            product_id.append(th.squeeze(p_id).detach().cuda().numpy())


    return np.concatenate(features, axis=0), np.concatenate(product_id)



backbone = open_clip.create_model_and_transforms('ViT-H-14', None)[0].visual
backbone.load_state_dict(th.load(path1))
backbone.half()
backbone.eval()


final_transform = get_final_transform()
img_dir = "/content/testing-dataset"
dataset_test = ProductDataset(img_dir, os.path.join(img_dir, "queries.csv"), None, final_transform, test_mode=True)
dataloader_test = DataLoader(dataset_test, batch_size=512, num_workers=4)

dataset_train = ProductDataset(img_dir, os.path.join(img_dir, "gallery.csv"), None, final_transform)
dataloader_train = DataLoader(dataset_train, batch_size=512, num_workers=4)

@th.no_grad()
def compute_score_test_data(model):
    embeddings_query, labels_query = extract_embeddings(model, dataloader_test, 1)
    embeddings_gallery, labels_gallery = extract_embeddings(model, dataloader_train, 1)

    _, indices = get_similiarity_l2(embeddings_gallery, embeddings_query, 1000)


    indices = indices.tolist()
    labels_gallery = labels_gallery.tolist()
    labels_query = labels_query.tolist()

    preds = convert_indices_to_labels(indices, labels_gallery)
    score = map_per_set(labels_query, preds)
    return score

model = backbone
model.cuda()

def predict(model):
    embeddings_query, labels_query = extract_embeddings(model, dataloader_test, 1)
    embeddings_gallery, labels_gallery = extract_embeddings(model, dataloader_train, 1)

    _, indices = get_similiarity_l2(embeddings_gallery, embeddings_query, 1000)


    indices = indices.tolist()
    labels_gallery = labels_gallery.tolist()
    labels_query = labels_query.tolist()

    preds = convert_indices_to_labels(indices, labels_gallery)
    score = map_per_set(labels_query, preds)
    return [indices, score]

[preds, score] = predict(model)
# Create a DataFrame from the predictions
df = pd.DataFrame(preds)

# Save the DataFrame to a CSV file
df.to_csv('preds.csv', index=False)






