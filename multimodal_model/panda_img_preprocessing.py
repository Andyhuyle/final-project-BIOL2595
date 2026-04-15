import tifffile
import torch
import numpy as np

def load_tiff_patches(path, patch_size=256, num_patches=10):
    img = tifffile.imread(path)  # shape: (H, W, C) or multi-layer
    
    if len(img.shape) == 3:
        H, W, C = img.shape
    else:
        H, W = img.shape[:2]
        C = 1
    
    patches = []
    
    for _ in range(num_patches):
        x = np.random.randint(0, H - patch_size)
        y = np.random.randint(0, W - patch_size)
        
        patch = img[x:x+patch_size, y:y+patch_size]
        patch = patch.astype(np.float32) / 255.0
        
        if C == 1:
            patch = np.expand_dims(patch, axis=-1)
        
        patch = np.transpose(patch, (2, 0, 1))  # CHW
        patches.append(patch)
    
    return torch.tensor(np.stack(patches))  # (N, C, H, W)

    
    