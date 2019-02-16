# define AdvMNIST : a dataset class that loads dataset from csv file
#
# csv should look like this: (comma seperated)
# image_path_name1.png,orig_label,adv_label
# image_path_name2.png,orig_label,adv_label
# etc...
# TBD: output of STN Network need to be saved in a folder and written in CSV file as specified above.

from torch.utils.data.dataset import Dataset
import pandas as pd
from PIL import Image
from torchvision import transforms
import numpy as np

# csv_path = 'adversarial_mnist_csv.csv'
class AdvMNIST(Dataset):
    def __init__(self, csv_path, transform=None):
        """
        Args:
            csv_path (string): path to csv file
            transform: pytorch transforms for transforms and tensor conversion
        """
        # Transforms
        self.transform = transform
        self.to_tensor = transforms.ToTensor()
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, header=None)
        # First column contains the image paths
        self.image_arr = np.asarray(self.data_info.iloc[:, 0])
        # Second column is the labels
        self.Origlabel_arr = np.asarray(self.data_info.iloc[:, 1])
        # Third column is for an operation indicator
        self.Advlabel_arr = np.asarray(self.data_info.iloc[:, 2])
        # Calculate len
        self.data_len = len(self.data_info.index)

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_arr[index]

        # Open image
        # with open(single_image_name) as single_im:
        single_im = Image.open(single_image_name)
        img_as_img = single_im.convert('L')
        # img_as_img = Image.open(single_image_name).convert('L')

        # Transform image to tensor
        img_as_tensor = self.to_tensor(img_as_img)
        # If there is an transform
        if self.transform is not None:
            img_as_tensor = self.transform(img_as_tensor)
        # Get label(class) of the image based on the cropped pandas column
        single_image_Origlabel = self.Origlabel_arr[index]
        single_image_Advlabel = self.Advlabel_arr[index]

        return img_as_tensor, single_image_Advlabel
        # return img_as_tensor, single_image_Origlabel, single_image_Advlabel

    def __len__(self):
        return self.data_len
