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

class GradientReversalFunc(Function):
	@staticmethod
	def forward(self, x, alpha):
		# Store context for backprop
		self.alpha = alpha

		# No operation during forward propagation 
		return x.view_as(x)

	@staticmethod
	def backward(self, grad_output):
		# Reverse the gradient by multiplying a negative
		output = grad_output.neg() * self.alpha

		# Must return same number as inputs to forward()
		return output, None

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

		self.domain_classifier = nn.Sequential(
			nn.Linear(self._to_linear, 100), nn.BatchNorm1d(100),
			nn.ReLU(True),
			nn.Linear(100, 6),
			nn.Softmax(dim=1)
		)

	def convs(self, x):
		x = self.feature_extractor(x)
		if self._to_linear is None:
			self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
		return x

	def forward(self, x, grl_lambda):
		features = self.feature_extractor(x)
		features = features.view(-1, self._to_linear)
		reverse_features = GradientReversalFunc.apply(features, grl_lambda)
		class_pred = self.classifier(features)
		domain_pred = self.domain_classifier(reverse_features)
		return class_pred, domain_pred

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
		if '/ADNI_' in img_path:
			LC_label = torch.Tensor([1, 0])
		elif '/AD_' in img_path:
			LC_label = torch.Tensor([0, 1])

		# Image label for domain classifier
		if 'siemens_3' in img_path:
			DC_label = torch.Tensor([1, 0, 0, 0, 0, 0])
		elif 'siemens_15' in img_path:
			DC_label = torch.Tensor([0, 1, 0, 0, 0, 0])
		elif 'philips_3' in img_path:
			DC_label = torch.Tensor([0, 0, 1, 0, 0, 0])
		elif 'philips_15' in img_path:
			DC_label = torch.Tensor([0, 0, 0, 1, 0, 0])
		elif 'GE_3' in img_path:
			DC_label = torch.Tensor([0, 0, 0, 0, 1, 0])
		elif 'GE_15' in img_path:
			DC_label = torch.Tensor([0, 0, 0, 0, 0, 1])

		if self.transform:
			image = self.transform(image)
		sample = {'image': image, 'label': LC_label, 'domain': DC_label}
		return sample

BATCH_SIZE = 256
file_path = '/home/justin.park/Summer_Research/dataset_creator/updated_dataset/training_dataset_path.csv'
dataset = CID(file_path, transform=transforms.Compose([transforms.Resize((224,224))]))
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

def fwd_pass(X, label_y, domain_y, grl_lambda=1, train=False):
	if train:
		model.zero_grad()

	class_pred, domain_pred = model(X, grl_lambda)

	class_matches = [torch.argmax(i) == torch.argmax(j) for i, j in zip(class_pred, label_y)]
	class_acc = class_matches.count(True)/len(class_matches)
	class_loss = class_loss_function(class_pred, label_y)

	domain_matches = [torch.argmax(i) == torch.argmax(j) for i, j in zip(domain_pred, domain_y)]
	domain_acc = domain_matches.count(True)/len(domain_matches)
	domain_loss = domain_loss_function(domain_pred, domain_y)

	loss = class_loss + domain_loss

	if train:
		loss.backward()
		optimizer.step()
	return class_acc, class_loss, domain_acc, domain_loss, loss

model = Net().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Loss function
class_loss_function = nn.MSELoss()
domain_loss_function = nn.MSELoss()

MODEL_NAME = f'model-{int(time.time())}'
print(MODEL_NAME)

def train():
	EPOCHS = 8
	counter = 0

	with open('model.log', 'a') as f:
		for epoch in range(EPOCHS):
			for data in tqdm(dataloader):
				counter += 1
				# grl_lambda can be specified to change the scalar value multiplied in the gradient reversal layer
				grl_lambda = 1
				batch_X = data['image'].to(device)
				batch_label_y = data['label'].to(device)
				batch_domain_y = data['domain'].to(device)
				class_acc, class_loss, domain_acc, domain_loss, loss = fwd_pass(batch_X.float(), batch_label_y, batch_domain_y, grl_lambda, train=True)

				if counter % 5 == 0:
					f.write(f'{MODEL_NAME}, {counter}, {round(float(class_acc),2)}, {round(float(class_loss),4)}, {round(float(domain_acc),2)}, {round(float(domain_loss),4)}, {round(float(loss),2)}, {epoch}\n')

# Train the model
train()

# Save model
model_path = 'VGG16_DA_Classifier.pt'
torch.save(model.state_dict(), model_path)
