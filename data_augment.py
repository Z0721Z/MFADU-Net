import os
from PIL import Image
import torchvision.transforms as transforms


def augment(original_folder, output_folder,time):

    data_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(), 
        #transforms.RandomVerticalFlip(),  
        #transforms.ColorJitter(brightness=0.4, contrast=0.5, saturation=0.5, hue=0.1),
    ])

    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(original_folder):
        if filename.endswith(".tif") or filename.endswith(".png") or filename.endswith(".gif"):

            image_path = os.path.join(original_folder, filename)
            image = Image.open(image_path)

            augmented_image = data_transform(image)

            file_parts = filename.split('.')
            file_parts=file_parts[0].split('_')
            file_parts[0] = file_parts[0] + '_augment'+time
            #file_parts[1] = 'png'
            name = file_parts[0]+'.png'
            print(name)

            output_path = os.path.join(output_folder, name)

            augmented_image.save(output_path)
            print(f"Augmented image saved: {output_path}")

def copy_images(original_folder,output_folder):
    for filename in os.listdir(original_folder):
        if filename.endswith(".tif") or filename.endswith(".png") or filename.endswith(".gif"):

            image_path = os.path.join(original_folder, filename)
            image = Image.open(image_path)
            file_parts = filename.split('.')
            file_parts[-1] = 'png'
            file1=file_parts[0].split('_')
            name =file1[0]+'_training.png'

            output_path = os.path.join(output_folder, name)
            image.save(output_path)
            print(f"Augmented image saved: {output_path}")

if __name__ == '__main__':
    original_folder = "./DRIVE/training/masks"
    output_folder = "./data/DRIVE/background"
    #augment(original_folder=original_folder,output_folder=output_folder,time='2')
    copy_images(original_folder=original_folder,output_folder=output_folder)
