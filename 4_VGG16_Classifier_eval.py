import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.autograd import Function
import time
import torch.nn as nn
from torchvision import models
import os
import numpy as np 
from tqdm import tqdm 
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
import pandas as pd
import nibabel as nib
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
import torchvision.transforms.functional as F


# GPU setup
if torch.cuda.is_available():
	device = torch.device("cuda:0")
	print("Run on GPU")
else:
	device = torch.device("cpu")
	print("Run on CPU")

# Convert 1 channel image to 3 channel
def grey_to_rgb(image):
	rgb_image = np.repeat(image[np.newaxis, ...], 3, 0)
	return rgb_image

# Model creation
class Net(nn.Module):

	def __init__(self):
		super().__init__()
		vgg16 = models.vgg16(pretrained=True)

		# Freeze up to the last 7th layer
		for param in vgg16.features[:-7].parameters():
			param.requires_grad = False

		self.feature_extractor = vgg16.features

		x = grey_to_rgb(torch.randn(224,224)).view(-1,3,224,224)

		self._to_linear = None
		self.convs(x)

		self.classifier = nn.Sequential(
			nn.Linear(self._to_linear, 100), nn.BatchNorm1d(100), nn.Dropout2d(),
			nn.ReLU(True),
			nn.Linear(100, 100), nn.BatchNorm1d(100),
			nn.ReLU(True),
			nn.Linear(100,2),
			nn.Softmax(dim=1)
		)

	def convs(self, x):
		x = self.feature_extractor(x)
		if self._to_linear is None:
			self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
		return x

	def forward(self, x):
		output = self.feature_extractor(x)
		output = output.view(-1, self._to_linear)
		output = self.classifier(output)
		return output

model = Net().to(device)

# Import model
model_path = 'VGG16_Classifier.pt'
model.load_state_dict(torch.load(model_path))
model.eval()

# Read nifti image files
def read_nifti_file(filepath):
	scan = nib.load(filepath)
	scan = scan.get_fdata()
	return scan

# Normalize image
def normalize(volume):
    volume = (volume - volume.min()) / (volume.max() - volume.min())
    volume = volume.astype("float32")
    return volume

def process_scan(path):
    volume = read_nifti_file(path)
    volume = normalize(volume)
    volume = grey_to_rgb(volume)
    return torch.Tensor(volume)

# Custom dataloader
class CID(Dataset):
	def __init__(self, annotations_file, transform=None):
		self.img_path = pd.read_csv(annotations_file, header=None)
		np.random.shuffle(self.img_path.values)
		self.transform = transform

	def __len__(self):
		return len(self.img_path)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		img_path = self.img_path.iloc[idx, 0]
		image = process_scan(img_path)

		# Image label for label classifier
		if '/CN_Update' in img_path:
			label = torch.Tensor([1, 0])
		elif '/AD_Update' in img_path:
			label = torch.Tensor([0, 1])

		if self.transform:
			image = self.transform(image)
		sample = {'image': image, 'label': label}
		return sample

BATCH_SIZE = 1
file_path = '/home/justin.park/Summer_Research/dataset_creator/updated_dataset/test_path.csv'
dataset = CID(file_path)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

font = {'family' : 'normal', 'size' : 20}
plt.rc('font', **font)
title_size = 25
size = 20

# Create confusion matrix to display results
def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=title_size)
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label', fontsize=size)
    plt.xlabel('Predicted label', fontsize=size)

# Test model
def test(net):
	class_names = ['Healthy', 'AD']

	class_pred_labels = []
	class_true_labels = []

	with torch.no_grad():
		for data in tqdm(dataloader):
			class_true_labels.append(class_names[torch.argmax(data['label']).to(device)])
			scan = data['image']
			scan = scan.squeeze(0)
			predictions = []

			height, width, depth = scan.shape

			# Test on 20 center slices of all 3 views for each subject
			image_1 = scan[round(height/2)-10:round(height/2)+10, :, :]
			image_2 = scan[:, round(width/2)-10:round(width/2)+10, :]
			image_3 = scan[:, :, round(depth/2)-10:round(depth/2)+10]

			for i in range(20):
				im_1 = image_1[i,:,:]
				im_1 = grey_to_rgb(im_1)
				im_1 = im_1.unsqueeze(0)
				im_1 = F.resize(im_1, (224,224))
				predictions.append(net(im_1.to(device)))

				im_2 = image_2[:,i,:]
				im_2 = grey_to_rgb(im_2)
				im_2 = im_2.unsqueeze(0)
				im_2 = F.resize(im_2, (224,224))
				predictions.append(net(im_2.to(device)))

				im_3 = image_3[:,:,i]
				im_3 = grey_to_rgb(im_3)
				im_3 = im_3.unsqueeze(0)
				im_3 = F.resize(im_3, (224,224))
				predictions.append(net(im_3.to(device)))

			predictions = torch.cat(predictions, dim=0)
			pred = torch.mean(predictions, 0, True)

			# Make the final prediction from the average
			class_pred_labels.append(class_names[torch.argmax(pred)])

	cm_class = confusion_matrix(y_true=class_true_labels, y_pred=class_pred_labels, labels=class_names)

	plot_confusion_matrix(cm=cm_class, classes=class_names, title='Class Results')
	plt.show()
	plt.savefig('Volume_class_results.png')

test(model)