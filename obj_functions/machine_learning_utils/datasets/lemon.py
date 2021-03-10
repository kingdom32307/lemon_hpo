from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import zipfile
import io
from tqdm import tqdm

np.random.seed(2021)

class ImageDataset(Dataset):
    def __init__(self, images, labels=None, transforms=None):
        '''
        images: List[PIL.Image, ...]
        labels: np.ndarray[np.int, ...]
        transforms: transforms.Compose
        '''
        super().__init__()
        self.images = images
        self.labels = labels
        self.transforms = transforms
        self.data_num = len(self.images)

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        out_images = self.images[idx]
        if self.transforms:
            out_images = self.transforms(out_images)
        if self.labels is None:
            return out_images
        else:
            out_labels = self.labels[idx]
            return out_images, out_labels 

def generate_image(files, data_zip):
    '''
    Args
    ----
    files: np.ndarray
    data_zip: String

    Returns
    -------
    imgs: List[PIL.Image, ...]
    '''
    with zipfile.ZipFile(data_zip) as img_zip:
        imgs = []
        for fname in tqdm(files):
            img = Image.open(io.BytesIO(img_zip.open(fname).read()))
            imgs.append(img)
    return imgs

def get_lemon(image_size = None):
    valid_size = 0.2

    train_csv = "./obj_functions/machine_learning_utils/datasets/train_images.csv"
    train_zip = "./obj_functions/machine_learning_utils/datasets/train_images.zip"
    train_dir = "train_images/"

    df_input = pd.read_csv(train_csv)
    df_input['file_path'] = train_dir + df_input.id.values

    labels = df_input.class_num.values
    
    # n_samples = len(labels)
    # indices = list(range(n_samples))
    # # np.random.shuffle(indice)
    # split = int(np.floor(valid_size * n_samples))
    # train_idx, valid_idx = indices[split:], indices[:split]
    skf = StratifiedKFold(5)
    train_idx, valid_idx = list(skf.split(labels, labels))[0]
    print(f"train / valid size: {len(train_idx)} / {len(valid_idx)}")
    df_valid = df_input.iloc[valid_idx, :].copy()
    df_train = df_input.iloc[train_idx, :].copy()

    # generate images, labels
    print('read train image')
    train_images = generate_image(df_train.file_path.values, train_zip)
    train_labels = df_train.class_num.values
    print('read valid image')
    valid_images = generate_image(df_valid.file_path.values, train_zip)
    valid_labels = df_valid.class_num.values


    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225])
    ])
    transform_valid = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225])
        ])


    trainset = ImageDataset(train_images, train_labels, transform_train)
    validset = ImageDataset(valid_images, valid_labels, transform_valid)

    train_dict = {
        "input": trainset,
        "train_idx": train_idx
    }

    valid_dict = {
        "input": validset,
        "valid_ixdx": valid_idx
    }

    return trainset, validset, train_idx, valid_idx

# class ImageFolder(Dataset):
#     IMG_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp"]

#     def __init__(self, img_dir, label_csv, transform=None):
#         dataframe = []
#         self.img_paths = self._get_img_paths(img_dir)
#         with open(label_csv, "r") as f:
#             for i, line in enumerate(f):
#                 if i == 0:
#                     continue
#                 row = line.strip().split(",")
#                 dataframe.append(row)

#         self.dataframe = dataframe
#         self.transform = transform
#         self.label_transform = transforms.ToTensor()
#         self.img_dir = img_dir


#     def __getitem__(self, index):
#         path = self.img_dir  + "/" + self.dataframe[index][0]
#         img = Image.open(path)
#         label = int(self.dataframe[index][1])

#         if self.transform is not None:
#             img = self.transform(img)

#         return img, label

#     def _get_img_paths(self, img_dir):
#         """指定したディレクトリ内の画像ファイルのパス一覧を取得する。
#         """
#         img_dir = Path(img_dir)
#         img_paths = [
#             p for p in img_dir.iterdir() if p.suffix in ImageFolder.IMG_EXTENSIONS
#         ]

#         return img_paths

#     def __len__(self):
#         """ディレクトリ内の画像ファイルの数を返す。
#         """
#         return len(self.img_paths)

# class MapSubset(Dataset):
#     """
#     Subset of a dataset at specified indices.

#     Arguments:
#         dataset (Dataset): The whole Dataset
#         indices (sequence): Indices in the whole set selected for subset
#     """
#     def __init__(self, dataset, map_fn):
#         self.dataset = dataset
#         self.map = map_fn

#     def __getitem__(self, index):
#         if self.map:     
#             data = self.map(self.dataset[index][0]) 
#         else:     
#             data = self.dataset[index][0]  # image
#         label = self.dataset[index][1]   # label      
#         return data, label 

#     def __len__(self):
#         return len(self.dataset)

# def get_lemon(image_size = None):
#     valid_size = 0.2

#     transform_train = transforms.Compose([
#         transforms.RandomResizedCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize(
#             mean=[0.485, 0.456, 0.406], 
#             std=[0.229, 0.224, 0.225])
#     ])
#     transform_valid = transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize(
#                 mean=[0.485, 0.456, 0.406], 
#                 std=[0.229, 0.224, 0.225])
#         ])

#     train_val_dataset = ImageFolder(
#                         img_dir = "/home/member/WORK/hpo_basement/obj_functions/machine_learning_utils/datasets/train_images",
#                         label_csv = "/home/member/WORK/hpo_basement/obj_functions/machine_learning_utils/datasets/train_images.csv"
#                         # transform=transform_train
#                         )

#     n_samples = len(train_val_dataset)
#     indices = list(range(n_samples))
#     np.random.shuffle(indices)
#     split = int(np.floor(valid_size * n_samples))
#     train_idx, val_idx = indices[split:], indices[:split]

#     train_dataset = Subset(train_val_dataset, train_idx)
#     val_dataset   = Subset(train_val_dataset, val_idx)
#     tng_data_tf = MapSubset(train_dataset, transform_train)
#     val_data_tf = MapSubset(val_dataset, transform_valid)
#     print(tng_data_tf)


#     return tng_data_tf, val_data_tf

# if __name__ == "__main__":
#     train_loader, train_labels, test_loader, val_labels, cls = get_lemon(image_size=32, test=True, all_train=False)

#     print(train_loader)
#     print(train_labels)