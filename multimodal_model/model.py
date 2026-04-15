import torch.nn as nn
import torchvision.models as models

class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(base.children())[:-1])
        self.fc = nn.Linear(512, 128)
    
    def forward(self, x):
        # x: (N_patches, C, H, W)
        feats = self.feature_extractor(x)  # (N, 512, 1, 1)
        feats = feats.view(feats.size(0), -1)
        feats = self.fc(feats)
        return feats.mean(dim=0)  # aggregate patches


class EHREncoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
    
    def forward(self, x):
        return self.net(x)


class MultimodalModel(nn.Module):
    def __init__(self, ehr_dim):
        super().__init__()
        self.img_enc = ImageEncoder()
        self.ehr_enc = EHREncoder(ehr_dim)
    
    def forward(self, patches, ehr):
        img_emb = self.img_enc(patches)
        ehr_emb = self.ehr_enc(ehr)
        
        return img_emb, ehr_emb