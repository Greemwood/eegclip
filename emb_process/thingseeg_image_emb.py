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
    
    features_filename = os.path.join(f'nonorm_thingseeg_{model_type}_features.pt')

    if train:
        img_directory = "THINGS-EEG/THINGS-EEG_images_set/training_images"  
    else:
        img_directory = "THINGS-EEG/THINGS-EEG_images_set/test_images"

    images = []
    all_folders = [d for d in os.listdir(img_directory) if os.path.isdir(os.path.join(img_directory, d))]
    all_folders.sort() 
    for folder in all_folders:
        folder_path = os.path.join(img_directory, folder)
        all_images = [img for img in os.listdir(folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
        all_images.sort()  
        images.extend(os.path.join(folder_path, img) for img in all_images)

    batch_size = 80  
    image_features_list = []
    
    for i in range(0, len(images), batch_size):
        print(i)
        batch_images = images[i:i + batch_size]
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
    img_emb(train=False)

            
    
        
    