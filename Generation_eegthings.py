import os

import torch
from torch.utils.data import DataLoader
os.environ["WANDB_API_KEY"] = "KEY"
os.environ["WANDB_MODE"] = 'offline'

import numpy as np
import torch.nn as nn
from eegdatasets_leaveone import EEGDataset
from Train_eegthings import EEGViT_pretrained,BrainDiffusionPrior
import random
class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class Proj_eeg(nn.Sequential):
    def __init__(self, embedding_dim=1440, proj_dim=1024, drop_proj=0.5):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                # nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )

class BDP(BrainDiffusionPrior):
    def __init__(self,prior_model=None,eeg_model=None,eeg_text_model=None,device='cuda'):
        super(BDP, self).__init__(prior_model,eeg_model,eeg_text_model,device)
    def generate(
            self, 
            c_embeds=None, 
            num_inference_steps=50, 
            timesteps=None,
            guidance_scale=5.0,
            generator=None
        ):
        # c_embeds (batch_size, cond_dim)
        self.prior_model.eval()
        N = c_embeds.shape[0] if c_embeds is not None else 1

        # 1. Prepare timesteps
        from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import retrieve_timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, self.device, timesteps)

        # 2. Prepare c_embeds
        if c_embeds is not None:
            c_embeds = c_embeds.to(self.device)

        # 3. Prepare noise
        # h_t = torch.randn(N, 1024, generator=generator, device=self.device)
        h_t = c_embeds

        # 4. denoising loop
        for _, t in enumerate(timesteps[35:]):
            t = torch.ones(h_t.shape[0], dtype=torch.float, device=self.device) * t
            # 4.1 noise prediction
            if guidance_scale == 0 or c_embeds is None:
                noise_pred = self.prior_model(h_t, t)
            else:
                noise_pred_cond = self.prior_model(h_t, t, c_embeds).predicted_image_embedding.squeeze(1)
                # noise_pred = noise_pred_cond
                attention_mask = torch.ones(N, 4, dtype=torch.float32,  device=self.device)
                attention_mask[:,:1] = 0.0
                noise_pred_uncond = self.prior_model(h_t, t, c_embeds, attention_mask= attention_mask).predicted_image_embedding
                # perform classifier-free guidance
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)


            # 4.2 compute the previous noisy sample h_t -> h_{t-1}
            h_t = self.scheduler.step(noise_pred, t.long().item(), h_t, generator=generator).prev_sample
        
        return h_t

class EEGViT(EEGViT_pretrained):
    def __init__(self,depth = 4,num_attention_heads=8):
        super(EEGViT,self).__init__(depth = depth, num_attention_heads=num_attention_heads)
        # super(EEGViT,self).__init__()
        self._initialize_weights()
    
    def forward(self,x):
        x = x.unsqueeze(1)
        x=self.conv1(x)
        # x=self.conv2(x)
        # x=self.batchnorm1(x)
        x=self.ViT.forward(x).pooler_output
        x = x.view(x.shape[0],-1)
        # x=self.ViT.forward(x,return_dict = False)
        x = self.proj_eeg1(x)
        # x_proj = self.proj_eeg2(x)
        # return  x, x_proj
        return  x


config = {
"data_path": "../THINGS-EEG/EEG/Preprocessed_data_512Hz",
}
seed = 3407
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

data_path = config['data_path']
emb_img_test = torch.load('no_norm_ViT-H-14_features_test.pt')
emb_img_train = torch.load('no_norm_ViT-H-14_features_train.pt')
#####################################################################################
eeg_model = EEGViT(depth = 4, num_attention_heads=8)
eeg_text_model = None
from diffusers import PriorTransformer
num_attention_heads = 16
diffusion_prior = PriorTransformer(num_attention_heads = num_attention_heads,
                                    attention_head_dim = 1024//num_attention_heads,
                                    embedding_dim=1024,
                                    encoder_hid_proj_type = None,
                                    num_embeddings = 1,
                                    additional_embeddings = 0,
                                    num_layers = 4,
                                    added_emb_type = "prd",
                                    dropout=0.2)
BrainPrior = BDP(eeg_model=eeg_model,eeg_text_model=eeg_text_model, prior_model=diffusion_prior, device=device)

sub = 'sub-10'
checkpoint_path = "models/contrast/Together/sub-10/eegclip_scale5_e200.pth"
checkpoint = torch.load(checkpoint_path)['model_state_dict']
dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
m, u = BrainPrior.load_state_dict(dict,strict=False)
print(f'missing keys:{m}')
print(f'unexpected keys:{u}')
BrainPrior.eval()
print('number of parameters:', sum([p.numel() for p in BrainPrior.parameters()]))

#####################################################################################

test_dataset = EEGDataset(data_path, subjects= [sub], train=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
text_features_test_all = test_dataset.text_features
img_features_test_all = test_dataset.img_features
train_dataset = EEGDataset(data_path, subjects= [sub], train=True,average = True)

import os

# Assuming generator.generate returns a PIL Image
from custom_pipeline import Generator4Embeds
generator = Generator4Embeds(num_inference_steps=4, device=device)

istest = True
if istest:
    directory = f"Generation/Generation/generated_imgs/{sub}"
else:
    directory = f"Generation/Generation/train_generated_imgs/{sub}"
with torch.no_grad():
    for k in range(200):
        if istest:
            # test_eegdata = test_dataset[k][0].unsqueeze(0).to(device)
            test_eegdata = test_dataset[k][0].unsqueeze(0).to(device)
            text = test_dataset[k][2]
        else:
            test_eegdata = train_dataset[k][0].unsqueeze(0).to(device)
            text = train_dataset[k][2]

        eeg_embeds = BrainPrior.eeg_model(test_eegdata)
        prompt_embeds = None
        text_prompt_embeds = None
        h = BrainPrior.generate(c_embeds=eeg_embeds, num_inference_steps=50, guidance_scale=20.0).to(dtype=torch.float16)
        text_prompt = ''
        blurred_image = None
        for j in range(1):
            image = generator.generate(h,text_prompt = text_prompt, prompt_embeds = prompt_embeds, blurred_image=blurred_image,text_prompt_embeds=text_prompt_embeds)
            if istest:
                path = f'{directory}/{text}/{j}.png'
            else:
                path = f'{directory}/{text}{k}/{j}.png'
            # Ensure the directory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)
            # Save the PIL Image
            image.save(path)
            print(f'Image saved to {path}')