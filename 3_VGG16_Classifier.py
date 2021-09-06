import torch
import torchvision.transforms as transforms
import torch.optim as optim
import time
import torch.nn as nn
from torchvision import models
import numpy as np 
from tqdm import tqdm 
from torch.utils.data import Dataset, DataLoader
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
			label = torch.Tensor([1, 0])
		elif '/AD_' in img_path:
			label = torch.Tensor([0, 1])
		if self.transform:
			image = self.transform(image)
		sample = {'image': image, 'label': label}
		return sample

BATCH_SIZE = 256
file_path = '/home/justin.park/Summer_Research/dataset_creator/updated_dataset/training_dataset_path.csv'
dataset = CID(file_path, transform=transforms.Compose([transforms.Resize((224,224))]))
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

def fwd_pass(X, y, train=False):
	if train:
		model.zero_grad()
	outputs = model(X)
	matches = [torch.argmax(i) == torch.argmax(j) for i, j in zip(outputs, y)]
	acc = matches.count(True)/len(matches)
	loss = loss_function(outputs, y)

	if train:
		loss.backward()
		optimizer.step()
	return acc, loss

model = Net().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Loss function
loss_function = nn.MSELoss()

MODEL_NAME = f'model-{int(time.time())}'
print(MODEL_NAME)

def train():
	
	EPOCHS = 8 			#Generally 5 to 10
	counter = 0

	with open('model.log', 'a') as f:
		for epoch in range(EPOCHS):
			for data in tqdm(dataloader):
				batch_X = data['image'].to(device)
				batch_y = data['label'].to(device)
				acc, loss = fwd_pass(batch_X.float(), batch_y, train=True)
				counter += 1

				if counter % 5 == 0:
					f.write(f'{MODEL_NAME}, {round(time.time(),3)}, {round(float(acc),2)}, {round(float(loss),4)}\n')

train()

# save model
model_path = 'VGG16_Classifier.pt'
torch.save(model.state_dict(), model_path)