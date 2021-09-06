import pandas as pd 
import os
from tqdm import tqdm
import nibabel as nib
import numpy as np

file_path = '/home/justin.park/Summer_Research/dataset_creator/updated_dataset/train_path.csv'
new_image_path = '/home/justin.park/Summer_Research/dataset_creator/updated_dataset/images/training_dataset_images'

img_paths = pd.read_csv(file_path, header=None)

def read_save_nifti_file(filepath, name):
	scan = nib.load(filepath)
	image = scan.get_fdata()
	image = np.squeeze(image)
	
	height, width, depth = image.shape

	image_1 = image[round(height/2)-10:round(height/2)+10, :, :]
	image_2 = image[:, round(width/2)-10:round(width/2)+10, :]
	image_3 = image[:, :, round(depth/2)-10:round(depth/2)+10]

	# Save 20 center slices of 3 different views for each subject
	for i in range(20):
		im_1 = image_1[i,:,:]
		im_2 = image_2[:,i,:]
		im_3 = image_3[:,:,i]

		filename_1 = name + '_' + filepath.split('/')[-1].split('.')[0] + str(i) + '1' + '.nii'
		filename_2 = name + '_' + filepath.split('/')[-1].split('.')[0] + str(i) + '2' + '.nii'
		filename_3 = name + '_' + filepath.split('/')[-1].split('.')[0] + str(i) + '3' + '.nii'

		im_1 = nib.Nifti1Image(im_1, scan.affine, scan.header)
		im_2 = nib.Nifti1Image(im_2, scan.affine, scan.header)
		im_3 = nib.Nifti1Image(im_3, scan.affine, scan.header)

		nib.save(im_1, os.path.join(new_image_path, filename_1))
		nib.save(im_2, os.path.join(new_image_path, filename_2))
		nib.save(im_3, os.path.join(new_image_path, filename_3))

for ind in tqdm(range(len(img_paths))):
	path = img_paths.iloc[ind, 0]
	if '/AD_Update' in path:
		if 'siemens_3' in path.lower():
			read_save_nifti_file(path, 'AD_siemens_3')
		if 'siemens_15' in path.lower():
			read_save_nifti_file(path, 'AD_siemens_15')
		if 'philips_3' in path.lower():
			read_save_nifti_file(path, 'AD_philips_3')
		if 'philips_15' in path.lower():
			read_save_nifti_file(path, 'AD_philips_15')
		if 'ge_3' in path.lower():
			read_save_nifti_file(path, 'AD_GE_3')
		if 'ge_15' in path.lower():
			read_save_nifti_file(path, 'AD_GE_15')
		
	if '/CN_Update' in path:
		if 'siemens_3' in path.lower():
			read_save_nifti_file(path, 'ADNI_siemens_3')
		if 'siemens_15' in path.lower():
			read_save_nifti_file(path, 'ADNI_siemens_15')
		if 'philips_3' in path.lower():
			read_save_nifti_file(path, 'ADNI_philips_3')
		if 'philips_15' in path.lower():
			read_save_nifti_file(path, 'ADNI_philips_15')
		if 'ge_3' in path.lower():
			read_save_nifti_file(path, 'ADNI_GE_3')
		if 'ge_15' in path.lower():
			read_save_nifti_file(path, 'ADNI_GE_15')
		


