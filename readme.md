## Environment setup

```
conda create -n eegclip python=3.10.14 -y 

conda activate eegclip

pip install -r requirements.txt
```

## Data availability

1.ThingsEEG

The raw and preprocessed EEG datasets, along with training and test images, are available on [OSF](https://osf.io/3jk45/). 

> **Note**: We downsampled the data to 512 Hz, which differs from the official release. 
>
> To use the dataset: 
>
> 1. Download the raw EEG data from the link above. 
>
> 2. Run the preprocessing script:   

```bash   
python EEG-preprocessing/preprocessing.py
```

2.Brain2Image dataset

The preprocessed EEG dataset, along with the training and test images, is available on [GitHub](https://github.com/perceivelab/eeg_visual_classification).

The preprocessed EEG dataset, the training and test images are available on [github](https://github.com/perceivelab/eeg_visual_classification).

Please use the following files:
- `block_splits_by_image_all.pth`
- `datasets/eeg_5_95_std.pth`

3.Image embedding

We provide image emb process code in /emb_process, or you can use our processed emb files. 

## Image Generation

If you want to quickly reproduce the results in the paper, please download the relevant `preprocessed data` and `model weights` from [Hugging Face](https://huggingface.co/sarahgreenwood/eegclip/tree/main).

```
#ThingsEEG dataset
python Generation_eegthings.py
#Brain2Image dataset
python Generation_brainiImage.py
```

Quantitive Metrics compute

```
#ThingsEEG dataset
/fMRI-reconstruction-NSD/src/Reconstruction_Metrics_ATM.ipynb
#Brain2Image dataset
/fMRI-reconstruction-NSD/src/Reconstruction_Metrics_ATM1.ipynb
```

## Training

```bash
#ThingsEEG dataset
export OMP_NUM_THREADS=1
torchrun --nproc_per_node=4 Train_eegthings.py
#Brain2Image dataset
export OMP_NUM_THREADS=1
torchrun --nproc_per_node=4 Train_brainiImage.py
```

## Acknowledge

1.Thanks to D Li et al. for their contribution in data set preprocessing and neural network structure, we refer to their work:
"[Visual Decoding and Reconstruction via EEG Embeddings with Guided Diffusion](https://arxiv.org/pdf/2403.07721)".
Dongyang Li, Chen Wei, Shiying Li, Jiachen Zou, Haoyang Qin, Quanying Liu1.

2.Thanks to R Yang et al. for their contribution in neural network structure, we refer to their work:
"[ViT2EEG: Leveraging Hybrid Pretrained Vision Transformers for EEG Data](https://arxiv.org/pdf/2308.00454)".
Ruiqi Yang, Eric Modesitt.

3.We also thank the authors of [Mindeye](https://github.com/alldbi/MindsEye) for providing quantitive metrics computing codes and the results. Thanks for the awesome research works.

4.Here we provide our THING-EEG dataset cited in the paper:
"[A large and rich EEG dataset for modeling human visual object recognition](https://www.sciencedirect.com/science/article/pii/S1053811922008758?via%3Dihub)".
Alessandro T. Gifford, Kshitij Dwivedi, Gemma Roig, Radoslaw M. Cichy.

5.Here we provide our Brain2Image dataset cited in the paper:
"[Decoding Brain Representations by Multimodal Learning of Neural Activity and Visual Features](https://arxiv.org/abs/1810.10974)".
S. Palazzo, C. Spampinato, I. Kavasidis, D. Giordano, J. Schmidt, M. Shah.



