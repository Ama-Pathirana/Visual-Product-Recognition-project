from get_image import *
from model_ensemble import *

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
model.cpu()

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