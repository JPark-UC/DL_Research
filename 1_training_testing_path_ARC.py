import os, csv
from glob import glob

train_name = 'train_path.csv'
test_name = 'test_path.csv'
counter = 0

paths = ['AD_Update', 'CN_Update']

with open(train_name, 'w', newline='') as new_train_file:
	train_w = csv.writer(new_train_file)
	with open(test_name, 'w', newline='') as new_test_file:
		test_w = csv.writer(new_test_file)
		for path in paths:
			img_path = os.path.join('/work/frayne_lab/vil/mariana/data/ADNI', path)
			for path, dirs, files, in os.walk(img_path):
				for file in files:
					if file.endswith('.nii') and not file.startswith('._'):
						img_path = os.path.join(path,file)
						if counter % 5 == 0 or 'T2' in path:
							test_w.writerow([img_path])
						else:
							train_w.writerow([img_path])
						counter += 1