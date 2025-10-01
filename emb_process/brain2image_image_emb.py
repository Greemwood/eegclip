import torch
import numpy as np
import os
import torch.nn as nn
from PIL import Image


device = "cuda:3" if torch.cuda.is_available() else "cpu"
model_type = 'ViT-H-14'
import open_clip
vlmodel, preprocess_train, feature_extractor = open_clip.create_model_and_transforms(
    model_type, pretrained ='laion/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin', precision='fp32', device = device)

import json

# Load the configuration from the JSON file
config_path = "data_config.json"
with open(config_path, "r") as config_file:
    config = json.load(config_file)

# Access the paths from the config
data_path = config["data_path"]

img_directory_training = config["img_directory_training"]
img_directory_test = config["img_directory_test"]

def img_emb(train=True):
    
    features_filename = os.path.join(f'nonorm_imagenet_{model_type}_features_new.pt') 
    loaded = torch.load('datasets/eeg_5_95_std.pth')

    imagenet = 'datasets/imageNet_images/'
    images = loaded["images"]

    image_path = [os.path.join(imagenet, image.split('_')[0], image+'.JPEG') for image in images]

    batch_size = 80  
    image_features_list = []
    
    for i in range(0, len(image_path), batch_size):
        print(i)
        batch_images = image_path[i:i + batch_size]
        image_inputs = torch.stack([preprocess_train(Image.open(img).convert("RGB")) for img in batch_images]).to(device)
        with torch.no_grad():
            batch_image_features = vlmodel.encode_image(image_inputs)
            # batch_image_features /= batch_image_features.norm(dim=-1, keepdim=True)
        image_features_list.append(batch_image_features)

    image_features = torch.cat(image_features_list, dim=0)

    # all_features_tensor = torch.cat((image_features),dim=0)

    torch.save({
        'img_features': image_features.cpu(),
    }, features_filename)

if __name__ == "__main__":
    img_emb(train=True)

            
    
        
    