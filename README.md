# 'SPARse-data augmented conTRAstive spatiotemporal embeddings’ (SPARTA)

This repository contains code and experiments for the paper titled: 'SPARse-data augmented conTRAstive spatiotemporal embeddings’ (SPARTA)

## Repository Structure

```
.
├── README.md
├── autoencoder_sampling
├── downstream_model_lstm_no_decoder/
├── latent_classification_model
├── simclr_no_cycle_loss_no_sampling
├── simclr_cycle_loss_no_sampling
├── simclr_no_cycle_loss_sampling
├── simclr_cycle_loss_sampling
├── simclr_multi_branch_gnn
├── simclr_multi_branch_self_attention
```

## Contents

- **autoencoder_sampling/**  
  Autoencoder Comparison - with Hard Negative Sampling Method.

- **downstream_model_lstm_no_decoder/**  
  LSTM-based downstream model for forecasting.

- **latent_classification_model/**  
  Latent Classification Model. 

- **latent_diffusion_model_conditional_attn/**  
  Conditional latent diffusion model. 

- **simclr_no_cycle_loss_no_sampling/**  
  Baseline SimCLR model - no cycle loss or hard negative sampling.

- **simclr_cycle_loss_no_sampling**  
  SimCLR model - with cycle loss, no hard negative sampling.

- **simclr_no_cycle_loss_sampling/**  
  SimCLR model - with hard negative sampling, no cycle loss.

- **simclr_cycle_loss_sampling/**  
  Baseline SimCLR model - with cycle loss and hard negative sampling.

- **simclr_multi_branch_gnn/**   
   Multimodal fusion SimCLR model, using GNN.

- **simclr_multi_branch_self_attention/**   
   Multimodal fusion SimCLR model, using Self-Attention.


## Getting Started

1. **Clone the repository**
    ```
    git clone <repo-url>
    ```

2. **Create and activate a virtual environment**
    ```
    python -m venv venv
    source venv/bin/activate
    ```

3. **Install dependencies**
    ```
    pip install -r requirements.txt
    ```

4. **Download ERA5 data using e.g. `python3 save_data.py` in the part 1 and part 2 directories.**

5. **Replace any paths in main.py to your local ERA5 Pytorch data file**
    e.g. replace `data = torch.load("/vol/bitbucket/nb324/ERA5_64x32_daily_850.pt")`

## Details on Folder Structure

Each model in the directories has corresponding files for training and testing. The main files of note are:

- main.py - **Main entry file for training and evaluating the model.**
- downstream_seed.py - **File to evaluate forecasting with strides of 5 and 10.**
- model.py / model_decoder.py - **Model code.**
- train.py / train_decoder.py - **Training code.**
- dataset.py - **Dataset code.**

Each model has several notebooks that provide results in the paper:

- eval_autoregressive.ipynb - **Evaluates the model for autoregressive forecasting**
- eval_autoregressive_seed_avg.ipynb -  **Evaluates the model for autoregressive forecasting for strided data**
- eval_latent.ipynb - **Evaluates the model for conditional latent diffusion**
- visual.ipynb - **Visualises the latent space and computes smoothness metrics**
- visual_window_next_t.ipynb - **Plots trajectories of context windows with the next step**

---
If you have any questions, please don't hesitate to contact me.
