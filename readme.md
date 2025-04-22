# SnapRender: Background Style Clustering for E-Commerce Product Images

## 📁 File Structure
SnapRender/
├── data/
│   └── train/
│       ├── bg1k_imgs/ # 1000 product images with full background
│       ├── bg1k_masks/ # corresponding foreground-only images
│       ├── bg60k_imgs_0/ # subsets of large dataset (0~4)
│       ├── bg60k_imgs_1/
│       ├── bg60k_imgs_2/
│       ├── bg60k_imgs_3/
│       ├── bg60k_imgs_4/
│       ├── bg60k_masks_0/ # Corresponding foreground-only images for above subsets
│       ├── bg60k_masks_1/
│       ├── bg60k_masks_2/
│       ├── bg60k_masks_3/
│       └── bg60k_masks_4/
│
├── models/
│   └── cvqvae.py # VQ-VAE model architecture (Encoder, Decoder, Quantizer)
│
├── utils/
│   ├── dataset_new.py # dataset loader
│   └── vq_loss.py # vector quantization loss implementation
│
├── samples_64_full/
│   ├── epoch1.png
│   ├── epoch30.png
│   ├── epoch60.png
│   ├── epoch90.png
│   ├── epoch100.png
│   └── vqvae_without_mask.pth # trained model weights
│
├── train_without_mask.ipynb # VQ-VAE training notebook (no masking, full image reconstruction)
├── style_cluster.ipynb # Extract latents, perform K-means, visualize with t-SNE
└── README.md # Project documentation (You are here!)

## ✅ Usage Notes
- Make sure your `data/train/` directory follows the correct structure (image and mask naming must match).
  Dataset can be downloaded at: https://github.com/Whileherham/BG60k
- All training and inference is running on GPU.
- Use `train_without_mask.ipynb` to train VQ-VAE model.
- Use `style_cluster.ipynb` to perform background style clustering and t-SNE visualization.