"""PyTorch dataset for CleanCam classification."""

from pathlib import Path

import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms

from cleancam_pipeline.core.constants import LABEL_TO_INDEX
from cleancam_pipeline.utils.image import image_to_pil


class CleanCamClassificationDataset(Dataset):
    """PyTorch dataset for CleanCam image classification."""

    def __init__(self, df: pd.DataFrame, transform: transforms.Compose):
        """
        Initialize dataset.

        Args:
            df: DataFrame with columns: absolute_path, label, image_id
            transform: Torchvision transforms to apply
        """
        self.df = df.reset_index(drop=True).copy()
        self.transform = transform

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.df)

    def __getitem__(self, idx: int):
        """
        Get a single sample.

        Returns:
            Tuple of (image_tensor, label_index, image_id)
        """
        row = self.df.iloc[idx]
        image = image_to_pil(Path(row["absolute_path"]))
        x = self.transform(image)
        y = LABEL_TO_INDEX[int(row["label"])]
        return x, y, str(row["image_id"])
