import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
# import faiss
import copy
import cv2
import time
import torchvision.transforms as T
import io
import open_clip
from PIL import Image

def get_similarity_brute_force(embeddings_gallery, embeddings_query, k):
    print('Processing indices...')

    s = time.time()
    distances = np.linalg.norm(embeddings_gallery - embeddings_query, axis=1)
    indices = np.argsort(distances)[:k]
    scores = distances[indices]
    e = time.time()

    print(f'Finished processing indices, took {e - s}s')
    return scores, indices


def get_similarity_l2(embeddings_gallery, embeddings_query, k):
    print('Processing indices...')

    s = time.time()
    dists = np.linalg.norm(embeddings_gallery - embeddings_query, axis=1)
    indices = np.argsort(dists)[:k]
    scores = dists[indices]
    e = time.time()

    print(f'Finished processing indices, took {e - s}s')
    return scores, indices

def get_similarity_IP(embeddings_gallery, embeddings_query, k):
    print('Processing indices...')

    s = time.time()
    dot_product = np.dot(embeddings_gallery, embeddings_query.T)
    norm_gallery = np.linalg.norm(embeddings_gallery, axis=1)
    norm_query = np.linalg.norm(embeddings_query)
    scores = dot_product / (norm_gallery * norm_query)
    indices = np.argsort(scores, axis=0)[-k:][::-1]
    e = time.time()

    print(f'Finished processing indices, took {e - s}s')
    return scores, indices


def convert_indices_to_labels(indices, labels):
    indices_copy = copy.deepcopy(indices)
    for row in indices_copy:
        for j in range(len(row)):
            row[j] = labels[row[j]]
    return indices_copy

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

def read_img(img_file, is_gray=False):
    img = Image.open(img_file)
    if is_gray:
        img = img.convert('L')
    else:
        img = img.convert('RGB')
    img = np.array(img)
    return img

def transform_img(image):
    img = image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if isinstance(img, np.ndarray):
        img =  Image.fromarray(img)

    transform = get_final_transform()
        
    img = transform(img)

    return img

@th.no_grad()
def extract_embeddings(model, image, epoch=10, use_cuda=False):
    features = []

    for _ in range(epoch):
        if use_cuda:
            image = image.cuda()

        # Ensure the input data type matches the weight data type
        features.append(model(image).detach().cpu().numpy().astype(np.float32))


    return np.concatenate(features, axis=0)




def Model():
    backbone = open_clip.create_model_and_transforms('ViT-H-14', None)[0].visual
    backbone.load_state_dict(th.load("./model1.pt"))
    # backbone.half()
    backbone.eval() 
    return backbone


def predict(image_data):
    image = np.array(image_data)
    image = transform_img(image).unsqueeze(0)

    model_1 = Model()

    embeddings_query = extract_embeddings(model_1, image, 1)
    embeddings_gallery = np.load("./embeddings_gallery.npy")
 

    _, indices = get_similarity_l2(embeddings_gallery, embeddings_query, 1000)

    indices = indices.tolist()

    return indices






