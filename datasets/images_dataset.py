import cv2
import jpeg4py as jpeg
from torch.utils.data import Dataset

from utils import data_utils


class ImagesDataset(Dataset):

    _TARGET_SIZE = 256

    def __init__(
        self,
        source_root,
        target_root,
        target_transform=None,
        source_transform=None,
        target_size=None,
    ):
        self.source_paths = sorted(data_utils.make_dataset(source_root))
        self.target_paths = sorted(data_utils.make_dataset(target_root))
        self.source_transform = source_transform
        self.target_transform = target_transform
        self.target_size = None
        if target_size is None:
            self.target_size = self._TARGET_SIZE
        else:
            self.target_size = target_size
        print("Target size:", self.target_size)

    def __len__(self):
        return len(self.source_paths)

    def __getitem__(self, index):
        from_path = self.source_paths[index]
        extension = from_path.split(".")[-1]
        if extension == 'jpg' or extension == 'jpeg':
            from_im = jpeg.JPEG(from_path).decode()
        else:

            from_im = cv2.imread(from_path)
            from_im = cv2.cvtColor(from_im, cv2.COLOR_BGR2RGB)
        from_im = cv2.resize(
            from_im,
            (self.target_size, self.target_size),
            interpolation=cv2.INTER_LINEAR,
        )

        to_path = self.target_paths[index]

        to_im = cv2.imread(to_path)
        to_im = cv2.cvtColor(to_im, cv2.COLOR_BGR2RGB)
        to_im = cv2.resize(
            to_im, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR
        )
        if self.target_transform:
            to_im = self.target_transform(to_im)

        if self.source_transform:
            from_im = self.source_transform(from_im)
        else:
            from_im = to_im

        return from_im, to_im
