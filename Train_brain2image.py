import os

import torch
import torch.optim as optim

# os.environ["WANDB_API_KEY"] = "KEY"
# os.environ["WANDB_MODE"] = 'offline'

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

from eegimagenetdataset import create_EEG_dataset

from loss import ClipLoss
import random
from utils import wandb_logger
import csv

import math
import traceback
from transformers import CLIPVisionModel, CLIPConfig, CLIPVisionModelWithProjection
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
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            # nn.LayerNorm(proj_dim),
        )
class Proj_eeg_proj(nn.Sequential):
    def __init__(self, embedding_dim=1024, proj_dim=1024, drop_proj=0.5):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )

class BrainDiffusionPrior(nn.Module):
    def __init__(self,prior_model=None,eeg_model=None,eeg_text_model=None,device='cuda',num_train_timesteps=1000):
        super(BrainDiffusionPrior, self).__init__()
        self.prior_model=prior_model
        self.prior_model.to(device)
        self.eeg_model=eeg_model
        self.eeg_model.to(device)
        # self.eeg_text_model = eeg_text_model
        # self.eeg_text_model.to(device)
        from diffusers.schedulers import DDPMScheduler
        self.scheduler = DDPMScheduler(num_train_timesteps = num_train_timesteps) 
        # self.scheduler = DDPMScheduler() 
        self.device = device
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.criterion = nn.MSELoss(reduction='none')
        self.i = 0
    def forward(self, eeg_data, img_features, alpha, imglabels):
        
        # print(self.i)
        self.i +=1
        h_embeds = img_features
        
        loss_img,eeg_features = self.eeg_model(eeg_data,img_features,alpha,imglabels)
        # loss_text,_ = self.eeg_text_model(eeg_data,img_features,text_features,alpha)
        c_embeds = eeg_features
        N = h_embeds.shape[0]

        # 1. randomly replecing c_embeds to None
        attention_mask = None
        if torch.rand(1) < 0.1:
            attention_mask = torch.ones(eeg_data.shape[0], 4, dtype=torch.float32,  device=self.device)
            attention_mask[:,:1] = 0.0
            # c_embeds = None
            # c_embeds = torch.zeros_like(h_embeds).to(h_embeds.dtype)

        # 2. Generate noisy embeddings as input
        noise = torch.randn_like(h_embeds)

        # 3. sample timestep
        timesteps = torch.randint(0, self.num_train_timesteps, (N,), device=self.device)

        # 4. add noise to h_embedding
        perturbed_h_embeds = self.scheduler.add_noise(
            h_embeds,
            noise,
            timesteps
        ) # (batch_size, embed_dim), (batch_size, )

        # 5. predict noise
        # perturbed_h_embeds = perturbed_h_embeds.unsqueeze(1)
        # c_embeds = c_embeds.unsqueeze(1)

        noise_pre = self.prior_model(perturbed_h_embeds, timesteps, c_embeds, attention_mask = attention_mask).predicted_image_embedding

        # 6. loss function weighted by sigma
        loss_prior = self.criterion(noise_pre.squeeze(1), noise) # (batch_size,)
        loss_prior = (loss_prior).mean()
        

        scale = 5
        loss = loss_img +  scale*loss_prior 
        return loss, loss_img.item(), scale*loss_prior.item(), eeg_features

import transformers
class EEGViT_pretrained(nn.Module):
    def __init__(self):
        super().__init__()
        # hidden_size = [128,1024]
        hidden_size = [32,256]
        conv2 = nn.Sequential(
            nn.Conv2d(
            in_channels=hidden_size[0], 
            out_channels=hidden_size[1],
            kernel_size=(1, 42),
            stride=(1, 8),
            padding=(0,0),
            bias=True
            ),
            # nn.AvgPool2d((1, 36), (1, 8)),
            nn.BatchNorm2d(hidden_size[1]),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        # Registering a scalar virtual parameter with a value of 1.0
        conv2.register_parameter('weight', nn.Parameter(torch.tensor(1.0)))
        conv1 = nn.Sequential(
            nn.Conv2d(
                    1, 
                    hidden_size[0], 
                    kernel_size=(128, 1), 
                    stride=(1, 1), 
                    padding=(0,0), 
                    bias=True),
            # nn.AvgPool2d((1, 36), (1, 8)),
            nn.BatchNorm2d(hidden_size[0]),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.conv1 = conv1
        # self.proj_eeg0 = Proj_eeg(embedding_dim=490, proj_dim=512)
        # self.conv2 = conv2
        model_name = "google/vit-base-patch16-224"
        config = transformers.ViTConfig.from_pretrained(model_name)
        config.update({'num_channels': hidden_size[0]})
        config.update({'attention_probs_dropout_prob': 0.1})
        config.update({'hidden_dropout_prob': 0.1})
        config.update({'image_size': (1,490)})
        config.update({'patch_size': (1,128)})
        config.update({'num_hidden_layers': 4})
        config.update({'hidden_size': hidden_size[1]})
        config.update({'num_attention_heads': 8})

        model = transformers.ViTModel(config=config)
        model.embeddings.patch_embeddings.projection = conv2
        model.embeddings.position_embeddings = nn.Parameter(torch.randn(1, 57 + 1, config.hidden_size))
        # model.embeddings.position_embeddings = nn.Parameter(torch.zeros(1, 32 + 1, config.hidden_size), requires_grad=False)
        self.ViT = model
        self.proj_eeg1 = Proj_eeg(embedding_dim=hidden_size[1],drop_proj=0.1)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func1 = ClipLoss()
        # self.loss_func2 = nn.MSELoss(reduction='none')
        self.not_initialize = []
        self._initialize_weights()
        # print(self.not_initialize)
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            else :
                self.not_initialize.append(m._get_name())
            
    def forward(self,x,img_features,alpha,imglabels):
        x = x.unsqueeze(1)

        x=self.conv1(x)

        pooler_output=self.ViT.forward(x).pooler_output

        x = self.proj_eeg1(pooler_output)


        logit_scale = self.logit_scale
        
        img_loss = self.loss_func1(x, img_features, logit_scale, imglabels)

        loss = img_loss
        return loss, x

          
def train_model(args, BrainPrior, dataloader, optimizer,lr_scheduler, device, text_features_all, img_features_all):
    BrainPrior.train()
    # img_model.train()
    # text_features_all = text_features_all.to(device).float() # (n_cls, d)
    img_features_all = (img_features_all[::10]).to(device).float()
    total_loss = 0
    total_loss_img = 0
    total_loss_prior = 0
    correct = 0
    total = 0
    alpha=0.9
    features_list = []  # List to store features
    save_features= True
    for batch_idx, (eeg_data, labels, img_features) in enumerate(dataloader):
        # print(batch_idx)
        eeg_data = eeg_data.to(device)
        # text_features = text_features.to(device).float()
        img_features = img_features.to(device).float()
        labels = labels.to(device)
        
        optimizer[0].zero_grad()
        optimizer[1].zero_grad()
        loss,loss_img, loss_prior, eeg_features_proj= BrainPrior(eeg_data, img_features, alpha, labels)
        features_list.append(eeg_features_proj)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(BrainPrior.parameters(), 1.0)
        optimizer[0].step()
        optimizer[1].step()
        # lr_scheduler[0].step(loss_img)
        lr_scheduler[1].step()
        total_loss += loss.item()
        total_loss_img += loss_img
        total_loss_prior += loss_prior
        # logits = logit_scale * eeg_features @ text_features_all.T # (n_batch, n_cls)
        if args.distributed:
            logit_scale = BrainPrior.module.eeg_model.logit_scale
        else:
            logit_scale = BrainPrior.eeg_model.logit_scale
        logits_img = logit_scale * eeg_features_proj @ img_features_all.T
        # logits_text = logit_scale * eeg_features @ text_features_all.T
        # logits_single = (logits_text + logits_img) / 2.0        
        # logits_text = logit_scale * eeg_features @ text_features_all.T
        logits_single = logits_img
        predicted = torch.argmax(logits_single, dim=1) # (n_batch, ) \in {0, 1, ..., n_cls-1}

        batch_size = predicted.shape[0]
        total += batch_size
        correct += (predicted == labels).sum().item()

    average_loss = total_loss / (batch_idx+1)
    average_loss_img = total_loss_img / (batch_idx+1)
    average_loss_prior = total_loss_prior / (batch_idx+1)
    accuracy = correct / total
    return average_loss, average_loss_img, average_loss_prior, accuracy, torch.cat(features_list, dim=0)

def evaluate_model(args, BrainPrior, dataloader, device, text_features_all, img_features_all, k):
    BrainPrior.eval()
    # img_model.eval()
    
    # text_features_all = text_features_all.to(device).float()
    img_features_all = img_features_all.to(device).float()
    total_loss = 0
    total_loss_img = 0
    total_loss_prior = 0
    correct = 0
    total = 0
    alpha = 0.9
    top5_correct = 0
    top5_correct_count = 0
    
    all_labels = set(range(img_features_all.size(0)))
    top5_acc = 0
    with torch.no_grad():
        for batch_idx, (eeg_data, labels,img_features) in enumerate(dataloader):
            # print(batch_idx)
            eeg_data = eeg_data.to(device)
            # text_features = text_features.to(device).float()
            labels = labels.to(device)
            img_features = img_features.to(device).float()

            loss, loss_img, loss_prior, eeg_features= BrainPrior(eeg_data, img_features, alpha, labels)
            # loss , eeg_features = eeg_model(eeg_data, img_features, text_features,alpha)
            if args.distributed:
                logit_scale = BrainPrior.module.eeg_model.logit_scale
            else:
                logit_scale = BrainPrior.eeg_model.logit_scale
            total_loss += loss.item()
            total_loss_img += loss_img
            total_loss_prior += loss_prior
            
            for idx, label in enumerate(labels):
                
                possible_classes = list(all_labels - {label.item()})
                selected_classes = random.sample(possible_classes, k-1) + [label.item()]
                selected_img_features = img_features_all[selected_classes]
                # selected_text_features = text_features_all[selected_classes]
                
                if k==200:
                    
                    logits_img = logit_scale * eeg_features[idx] @ selected_img_features.T
                    logits_single = logits_img
                    # print("logits_single", logits_single.shape)
                    
                    # predicted_label = selected_classes[torch.argmax(logits_single).item()]
                    predicted_label = selected_classes[torch.argmax(logits_single).item()] # (n_batch, ) \in {0, 1, ..., n_cls-1}
                    if predicted_label == label.item():
                        # print("predicted_label", predicted_label)
                        correct += 1
                    
                    
                    
                    
                    # print("logits_single", logits_single)
                    _, top5_indices = torch.topk(logits_single, 5, largest =True)
                                                           
                    
                    if label.item() in [selected_classes[i] for i in top5_indices.tolist()]:                
                        top5_correct_count+=1                                
                    total += 1
                elif k == 50 or k == 100:
                    
                    selected_classes = random.sample(possible_classes, k-1) + [label.item()]

                    logits_img = logit_scale * eeg_features[idx] @ selected_img_features.T
                    logits_single = logits_img
                    
                    predicted_label = selected_classes[torch.argmax(logits_single).item()]
                    if predicted_label == label.item():
                        correct += 1
                    _, top5_indices = torch.topk(logits_single, 5, largest =True)
                                                           
                    
                    if label.item() in [selected_classes[i] for i in top5_indices.tolist()]:                
                        top5_correct_count+=1                                
                    total += 1
                elif k==2 or k==4 or k==10:
                    selected_classes = random.sample(possible_classes, k-1) + [label.item()]
                    
                    logits_img = logit_scale * eeg_features[idx] @ selected_img_features.T
                    # logits_text = logit_scale * eeg_features[idx] @ selected_text_features.T
                    # logits_single = (logits_text + logits_img) / 2.0
                    logits_single = logits_img
                    # print("logits_single", logits_single.shape)
                    
                    # predicted_label = selected_classes[torch.argmax(logits_single).item()]
                    predicted_label = selected_classes[torch.argmax(logits_single).item()] # (n_batch, ) \in {0, 1, ..., n_cls-1}
                    if predicted_label == label.item():
                        correct += 1
                    total += 1
                else:
                    print("Error.")
                    
    average_loss = total_loss / (batch_idx+1)
    average_loss_img = total_loss_img / (batch_idx+1)
    average_loss_prior = total_loss_prior / (batch_idx+1)
    accuracy = round(correct / total, 3)
    top5_acc = round(top5_correct_count / total, 3)
    return average_loss, average_loss_img, average_loss_prior, accuracy, top5_acc

def main_train_loop(sub, epoch0, current_time, BrainPrior, train_dataloader, test_dataloader, optimizer, lr_scheduler, device, 
                    text_features_train_all, text_features_test_all, img_features_train_all, img_features_test_all, config, args = None, logger=None,outdir=None ):
    logger = wandb_logger(config) if logger else None
    logger.watch(BrainPrior,logger) 
    # logger.watch(img_model,logger) 
    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []
    test_losses_img, test_losses_prior = [], []
    v2_accs = []
    v4_accs = []
    v10_accs = []

    best_accuracy = 0.0
    best_model_weights = None
    best_epoch_info = {}
    results = []  # List to store results for each epoch
    from diffusers.optimization import get_cosine_schedule_with_warmup
    # lr_scheduler = get_cosine_schedule_with_warmup(
    #         optimizer=optimizer,
    #         num_warmup_steps=500,
    #         num_training_steps=(len(train_dataloader) * config['epochs']),
    #     )
    for epoch in range(epoch0, config['epochs']):
        if args.distributed:
            train_dataloader.sampler.set_epoch(epoch)
        
        train_loss, average_loss_img, average_loss_prior, train_accuracy, features_tensor = train_model(args,BrainPrior, train_dataloader, optimizer,lr_scheduler, device, text_features_train_all, img_features_train_all)
        # lr_scheduler[0].step(average_loss_img)
        # train_loss, train_accuracy, features_tensor = 0,0,0
        test_loss, test_loss_img, test_loss_prior, test_accuracy, top5_acc = evaluate_model(args,BrainPrior,  test_dataloader, device, text_features_test_all, img_features_test_all,k=100)
        lr_scheduler[0].step(test_loss_img)
        _,_,_, v2_acc, _ = evaluate_model(args,BrainPrior,  test_dataloader, device, text_features_test_all, img_features_test_all, k = 2)
        _,_,_, v4_acc, _ = evaluate_model(args,BrainPrior,  test_dataloader, device, text_features_test_all, img_features_test_all, k = 4)
        _,_,_, v10_acc, _ = evaluate_model(args,BrainPrior,  test_dataloader, device, text_features_test_all, img_features_test_all, k = 10)
        _,_,_, v50_acc, v50_top5_acc = evaluate_model(args,BrainPrior,  test_dataloader, device, text_features_test_all, img_features_test_all,  k=50)
        _,_,_, v100_acc, v100_top5_acc = evaluate_model(args,BrainPrior,  test_dataloader, device, text_features_test_all, img_features_test_all,  k=100)
        test_losses.append(test_loss)
        test_losses_img.append(test_loss_img)
        test_losses_prior.append(test_loss_prior)
        test_accuracies.append(test_accuracy)
        v2_accs.append(v2_acc)
        v4_accs.append(v4_acc)
        v10_accs.append(v10_acc)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        # Append results for this epoch
        epoch_results = {
        "epoch": epoch + 1,
        # "train_loss": train_loss,
        # "train_accuracy": train_accuracy,
        "test_loss": test_loss,
        "test_loss_img": test_loss_img,
        "test_loss_prior": test_loss_prior,
        "test_accuracy": test_accuracy,
        "v2_acc": v2_acc,
        "v4_acc": v4_acc,
        "v10_acc": v10_acc,
        "top5_acc":top5_acc,
        "v50_acc": v50_acc,
        "v100_acc": v100_acc,
        "v50_top5_acc":v50_top5_acc,
        "v100_top5_acc": v100_top5_acc
        }

        results.append(epoch_results)
        
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            # best_model_weights = model.state_dict().copy()
            
            best_epoch_info = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
                "v2_acc":v2_acc,
                "v4_acc":v4_acc,
                "v10_acc":v10_acc
            }
        logger.log({
            "Train Loss": train_loss,
            "Loss_img": average_loss_img,
            "Loss_prior": average_loss_prior,
            "Train Accuracy": train_accuracy,
            "Test Loss": test_loss,
            "test_loss_img": test_loss_img,
            "test_loss_prior": test_loss_prior,
            "Test Accuracy": test_accuracy,
            "v2 Accuracy": v2_acc,
            "v4 Accuracy": v4_acc,
            "v10 Accuracy": v10_acc,
            "Epoch": epoch
        })

        print(f"Epoch {epoch + 1}/{config['epochs']} - Train Loss: {train_loss:.4f}, Loss_img: {average_loss_img:.4f}, Loss_prior: {average_loss_prior:.4f}, Train Accuracy: {train_accuracy:.4f}, Learning rate: {optimizer[0].param_groups[0]['lr']:.8f}")
        print(f"Epoch {epoch + 1}/{config['epochs']} - Test Loss: {test_loss:.4f}, Test Loss img: {test_loss_img:.4f}, Test Loss prior: {test_loss_prior:.4f}, Test Accuracy: {test_accuracy:.4f}, Top5 Accuracy: {top5_acc:.4f}")
        print(f"Epoch {epoch + 1}/{config['epochs']} - v2 Accuracy:{v2_acc} - v4 Accuracy:{v4_acc} - v10 Accuracy:{v10_acc} - v50 Accuracy:{v50_acc} - v100 Accuracy:{v100_acc}")
        if (epoch +1) % 100 == 0:                    
            if args.rank==0:
                if config['insubject']==True:       
                    # os.makedirs(f"./models/contrast/{config['encoder_type']}/{sub}/{current_time}", exist_ok=True)             
                    # file_path = f"./models/contrast/{config['encoder_type']}/{sub}/{current_time}/{epoch+1}.pth"
                    # torch.save(BrainPrior.state_dict(), file_path)     
                    file_path = os.path.join(outdir, f'{epoch+1}.pth') 
                    state_dict = BrainPrior.state_dict()
                    for key in list(state_dict.keys()):
                        if 'module.' in key:
                            state_dict[key.replace('module.', '')] = state_dict[key]
                            del state_dict[key]
                    try:
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': state_dict,
                            'optimizer_state_dict': [optimizer[0].state_dict(),optimizer[1].state_dict()],
                            'train_losses': train_losses,
                            'val_losses': test_losses_img,
                            # 'val_losses': val_losses,
                            # 'lrs': lrs,
                            }, file_path)   
                    except:
                        print('Failed to save weights')
                        print(traceback.format_exc())    
                else:                
                    # os.makedirs(f"./models/contrast/across/{config['encoder_type']}/{current_time}", exist_ok=True)
                    file_path = os.path.join(outdir, f'{epoch+1}.pth')        
                    # file_path = f"./models/contrast/across/{config['encoder_type']}/{current_time}/{epoch+1}.pth"
                    # torch.save(BrainPrior.state_dict(), file_path)
                    state_dict = BrainPrior.state_dict()
                    for key in list(state_dict.keys()):
                        if 'module.' in key:
                            state_dict[key.replace('module.', '')] = state_dict[key]
                            del state_dict[key]
                    try:
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': state_dict,
                            'optimizer_state_dict': optimizer.state_dict(),
                            'train_losses': train_losses,
                            'val_losses': test_losses_img,
                            # 'lrs': lrs,
                            }, file_path)   
                    except:
                        print('Failed to save weights')
                        print(traceback.format_exc()) 
                print(f"model saved in {file_path}!")


    
        # model.load_state_dict(best_model_weights)

        
        # torch.save(model.state_dict(), '{train_pos_img_text}.pth')

    if args.rank == 0:
        fig, axs = plt.subplots(3, 2, figsize=(10, 15))

        
        axs[0, 0].plot(train_losses, label='Train Loss')
        axs[0, 0].plot(test_losses, label='Test Loss')
        axs[0, 0].legend()
        axs[0, 0].set_title("Loss Curve")

        
        axs[0, 1].plot(train_accuracies, label='Train Accuracy')
        axs[0, 1].plot(test_accuracies, label='Test Accuracy')
        axs[0, 1].legend()
        axs[0, 1].set_title("Accuracy Curve")

        
        
        axs[1, 0].plot(v2_accs, label='2-class Accuracy')
        axs[1, 0].legend()
        axs[1, 0].set_title("2-Class Accuracy Curve")

        
        axs[1, 1].plot(v4_accs, label='4-class Accuracy')
        axs[1, 1].legend()
        axs[1, 1].set_title("4-Class Accuracy Curve")

        
        axs[2, 0].plot(v10_accs, label='10-class Accuracy')
        axs[2, 0].legend()
        axs[2, 0].set_title("10-Class Accuracy Curve")

        
        info_text = (f"Best Model Info (from Epoch {best_epoch_info['epoch']}):\n"
                    f"Train Loss: {best_epoch_info['train_loss']:.4f}\n"
                    f"Train Accuracy: {best_epoch_info['train_accuracy']:.4f}\n"
                    f"Test Loss: {best_epoch_info['test_loss']:.4f}\n"
                    f"Test Accuracy: {best_epoch_info['test_accuracy']:.4f}\n"
                    f"v2_acc:{best_epoch_info['v2_acc']:.4f}\n"
                    f"v4_acc:{best_epoch_info['v4_acc']:.4f}\n"
                    f"v10_acc:{best_epoch_info['v10_acc']:.4f}")

        axs[2, 1].axis('off')  
        axs[2, 1].text(0.5, 0.5, info_text, fontsize=10, ha='center', va='center', transform=axs[2, 1].transAxes)

        plt.tight_layout()

        
        plt.suptitle('pos_img_text', fontsize=16, y=1.05)
        
        plt.savefig(os.path.join(outdir,'pos_img_text'))
        logger.finish()
    return results

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        args.rank = 0
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)
def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0,
                     start_warmup_value=0, warmup_steps=-1):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array(
        [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule

import datetime
import argparse
import torch.distributed as dist

def main():  
    config = {
        # "data_path": "THINGS-EEG/EEG/Preprocessed_data_200Hz",
        "data_path":"THINGS-EEG/EEG/Preprocessed_data_512Hz",
        "project": "brain-image",
        # "entity": "ibrain_bci",
        "name": "lr=1e-3_img_pos_pro_eeg",
        "lr1": 3e-4,
        "lr2":5e-4,
        # "min_lr":1e-5,
        "epochs": 1500,
        "warmup_epochs":5,
        "batch_size": 64,
        "logger": True,
        "insubject": True,
        "encoder_type": 'Together',
        "distributed": 'False'
    }
    parser = argparse.ArgumentParser('ATM Training', add_help=False)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--seed', default=3407, type=int)
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    args = parser.parse_args([])
    init_distributed_mode(args)
    print(args)
    # num_devices = int(os.environ.get("NUM_GPUS",4))
    # print('num_devices:', num_devices)
    num_devices = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_devices}")

    device = torch.device("cuda")
    if args.distributed:
        seed = args.seed + dist.get_rank()
    else:
        seed = args.seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.benchmark = True
    data_path = config['data_path']
    # subjects = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07', 'sub-08', 'sub-09']
    subjects = [1]
    # subjects = [2,3,5,6,1]

    current_time = datetime.datetime.now().strftime("%m-%d_%H-%M")  
    # splits_path = "datasets/block_splits_by_image_single.pth"#++
    splits_path = "../datasets/block_splits_by_image_all.pth"#++
    eeg_signals_path="../datasets/eeg_5_95_std.pth"#++
    # eeg_signals_path="datasets/eeg_signals_raw_with_mean_std.pth"#++
    # eeg_signals_path="datasets/eeg_55_95_std.pth"#++
    for timesteps in [1000]:
        for sub in subjects:   
            train_dataset, val_dataset, test_dataset = create_EEG_dataset(eeg_signals_path = eeg_signals_path, splits_path = splits_path, 
                subject = sub)                 

            num_train = len(train_dataset)
            # trainfeature66160=train_dataset[66160]
            # testfeature150=test_dataset[150]
            aa = train_dataset[10]
            if  args.distributed:
                num_tasks = dist.get_world_size()
                global_rank = dist.get_rank()
                sampler_train = torch.utils.data.DistributedSampler(
                    train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
                )
                sampler_val = torch.utils.data.DistributedSampler(
                    val_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False
                )
                print("Sampler_train = %s" % str(sampler_train))
                sampler_test = torch.utils.data.DistributedSampler(
                    test_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False
                )
            else:
                global_rank = 0
                sampler_train = torch.utils.data.RandomSampler(train_dataset)
                sampler_val = None 
                sampler_test = None
            train_loader = torch.utils.data.DataLoader(
                train_dataset, sampler=sampler_train,
                batch_size=config['batch_size'],
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=False,
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset, sampler=sampler_val,
                batch_size=int(1.5 * config['batch_size']),
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=False,
            )
            test_loader = torch.utils.data.DataLoader(
                    test_dataset, sampler=sampler_test,
                    batch_size=int(1.5 * config['batch_size']),
                    num_workers=args.num_workers,
                    pin_memory=args.pin_mem,
                    drop_last=False
                )
            text_features_train_all = train_dataset.text_features
            # text_features_test_all = test_dataset.text_features
            img_features_train_all = train_dataset.img_features
            # img_features_test_all = test_dataset.img_features

            text_features_test_all = val_dataset.text_features
            img_features_test_all = val_dataset.img_features


            # Re-initialize the models for each subject

            # eeg_model = globals()[config['encoder_type']](sequence_length=512)
            eeg_model = EEGViT_pretrained()
            eeg_text_model = None
            # eeg_text_model = EEGtext_pretrained()

            from diffusers import PriorTransformer, UNet2DConditionModel
            num_attention_heads = 16
            diffusion_prior = PriorTransformer(num_attention_heads = num_attention_heads,
                                                attention_head_dim = 1024//num_attention_heads,
                                                embedding_dim=1024,
                                                encoder_hid_proj_type = None,
                                                num_embeddings = 1,
                                                additional_embeddings = 0,
                                                num_layers = 4,
                                                added_emb_type = "prd",
                                                dropout=0.1)
            # number of parameters
            print(sum(p.numel() for p in diffusion_prior.parameters() if p.requires_grad))
            BrainPrior = BrainDiffusionPrior(eeg_model=eeg_model,eeg_text_model=eeg_text_model, prior_model=diffusion_prior, device=device, num_train_timesteps = timesteps)
            BrainPrior.to(device)
            model = BrainPrior
            if args.distributed:
                model = torch.nn.parallel.DistributedDataParallel(BrainPrior, device_ids=[args.gpu], find_unused_parameters=True)
                model_without_ddp = model.module
                groups = [{'params': model_without_ddp.eeg_model.parameters(), 'lr': config['lr1']},
                        {'params': model_without_ddp.prior_model.parameters(), 'lr': config['lr2']}]
            else:
                groups = [{'params': model.eeg_model.parameters(), 'lr': config['lr1']},
                        {'params': model.prior_model.parameters(), 'lr': config['lr2']}]

            # if args.distributed:
            #     total_batch_size = config['batch_size']  * dist.get_world_size()
            # num_training_steps_per_epoch = len(train_dataset) // total_batch_size

            
            # optimizer = torch.optim.AdamW(itertools.chain(eeg_model.parameters(), img_model.parameters()), lr=config['lr'])  

            # optimizer = torch.optim.AdamW(groups)  
            optimizer_eeg = optim.AdamW([groups[0]])
            optimizer_prior = optim.AdamW([groups[1]])
            lr_scheduler_eeg = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_eeg, mode='min', factor=0.9, 
                                                              patience=10, threshold=0.01, 
                                                              threshold_mode='rel', cooldown=0, 
                                                              min_lr=1e-7, eps=1e-08, verbose=True )
            
            # lr_scheduler_prior = torch.optim.lr_scheduler.CyclicLR(optimizer_prior, base_lr=0.1*config['lr2'], 
            #                                                                    max_lr=config['lr2'], step_size_up=1000, 
            #                                                                    mode='triangular', 
            #                                                                    cycle_momentum=False)
            lr_scheduler_prior = torch.optim.lr_scheduler.OneCycleLR(optimizer_prior, max_lr=config['lr2'], 
                                                total_steps=config['epochs']*math.ceil(num_train/config['batch_size']/num_devices), 
                                                div_factor=10,
                                                final_div_factor=10,
                                                last_epoch=-1, pct_start=0.5)  

            optimizer = [optimizer_eeg,optimizer_prior]
            lr_scheduler = [lr_scheduler_eeg, lr_scheduler_prior]
  

            outdir = f"./models/contrast/{config['encoder_type']}/{sub}/{current_time}"
            # outdir = f"./models/contrast/{config['encoder_type']}/{sub}/Final1"
            
            # outdir = 'models/contrast/Together/2/09-25_23-36'
            if args.rank==0:
                os.makedirs(outdir, exist_ok=True)
                current_script_path = os.path.abspath(__file__)
                import shutil
                shutil.copy(current_script_path, outdir)
                shutil.copy('Generation_brain2image.py', outdir)
            resume_from_ckpt = False
            if os.path.exists(os.path.join(outdir, '1000.pth')):
                ckpt_path = os.path.join(outdir, '1000.pth')
                resume_from_ckpt = True
            if resume_from_ckpt:
                print("\n---resuming from ckpt_path---\n", ckpt_path)
                checkpoint = torch.load(ckpt_path, map_location=device)
                epoch = checkpoint['epoch']+1
                optimizer[0].load_state_dict(checkpoint['optimizer_state_dict'][0]) 
                optimizer[1].load_state_dict(checkpoint['optimizer_state_dict'][1]) 
                if hasattr(model, 'module'):
                    model.module.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint['model_state_dict'])
                global_batch_size = config['batch_size'] * num_devices
                total_steps_done = epoch*((num_train//config['batch_size'])//num_devices)
                for _ in range(total_steps_done):
                    lr_scheduler[1].step()
                    # lr_scheduler[1].step()
                # lr_scheduler[0]['lr'] = optimizer[0]['param_groups']['lr']
                del checkpoint
                torch.cuda.empty_cache()
            else:
                epoch = 0
            print(f'Processing {sub}: number of parameters:', sum([p.numel() for p in model.parameters()]))
            import time
            start_time = time.time()
            results = main_train_loop(sub, epoch, current_time, model,  train_loader, val_loader,  optimizer, lr_scheduler, device, text_features_train_all, text_features_test_all, img_features_train_all, img_features_test_all, config, args, logger=config['logger'],outdir = outdir)
            
            if global_rank == 0:
                total_time = time.time() - start_time
                total_time_str = str(datetime.timedelta(seconds=int(total_time)))
                print('Training time {}'.format(total_time_str))
                # Save results to a CSV file
                # results_dir = f"./outputs/contrast/{config['encoder_type']}/{sub}/{current_time}"
                results_dir = outdir
                os.makedirs(results_dir, exist_ok=True)          
                results_file = f"{results_dir}/{config['encoder_type']}_{'cross_exclude_' if not config['insubject'] else ''}{sub}.csv"
                with open(results_file, 'w', newline='') as file:
                    writer = csv.DictWriter(file, fieldnames=results[0].keys())
                    writer.writeheader()
                    writer.writerows(results)
                    # writer.writerows(total_time)
                print(f'Results saved to {results_file}')

if __name__ == '__main__':
    main()