from torch.utils.data import Dataset
import cv2


class Pix2PiXHD_Dataset(Dataset):
    """Pix2PixHD image dataset.

    Parameters
    -----------------------------
    root: pathlib.PosixPath
        data dir
    transfor: torchvision.transform.Compose
    """

    def __init__(
        self,
        root,
        transform=None,
        image_reader=cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB),
        val_size=None,
    ):
        if not root.is_absolute():
            self.abs_data_dir = root.resolve()
        else:
            self.abs_data_dir = root
        if val_size:
            self.data_path = [
                path
                for i, path in enumerate(self.abs_data_dir.glob("*.jpg"))
                if i < val_size
            ]
        else:
            self.data_path = [path for path in self.abs_data_dir.glob("*.jpg")]
        self.loader = loader
        self.spliter = spliter
        self.normalizer = normalizer
        self.transform = transform

    def __getitem__(self, idx):
        image = self.loader(self.data_path[idx])
        input_A, output_B = self.spliter(image)

        if self.transform:
            input_A, output_B = self.transform(input_A, output_B)

        # apply normalization to output_B only
        if self.normalizer:
            return input_A, self.normalizer(output_B)
        return input_A, output_B

    def __len__(self):
        return len(self.data_path)
