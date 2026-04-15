from torch.utils.data import Dataset
import random

class MultimodalDataset(Dataset):
    def __init__(self, ehr_features, labels_df, image_dir):
        self.ehr = ehr_features
        self.labels = labels_df
        self.image_dir = image_dir
        
        # group images by ISUP grade
        self.isup_groups = {}
        for i, row in labels_df.iterrows():
            g = row["isup_grade"]
            self.isup_groups.setdefault(g, []).append(row)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        row = self.labels.iloc[idx]
        isup = row["isup_grade"]
        
        # load image
        img_path = f"{self.image_dir}/{row['image_id']}.tiff"
        patches = load_tiff_patches(img_path)
        
        # sample random EHR (weak alignment)
        ehr_idx = random.randint(0, len(self.ehr) - 1)
        ehr_vec = torch.tensor(self.ehr[ehr_idx]).float()
        
        return patches, ehr_vec, isup