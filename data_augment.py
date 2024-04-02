import os
from PIL import Image
import torchvision.transforms as transforms


def augment(original_folder, output_folder,time):
    # 定义数据增强操作
    data_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # 水平翻转
        #transforms.RandomVerticalFlip(),  # 垂直翻转
        #transforms.ColorJitter(brightness=0.4, contrast=0.5, saturation=0.5, hue=0.1),  # 调整亮度、对比度、饱和度和色调
    ])
    # 创建保存增强图像的文件夹
    os.makedirs(output_folder, exist_ok=True)
    # 遍历原始图像文件夹中的所有图像文件
    for filename in os.listdir(original_folder):
        if filename.endswith(".tif") or filename.endswith(".png") or filename.endswith(".gif"):
            # 加载原始图像
            image_path = os.path.join(original_folder, filename)
            image = Image.open(image_path)
            # 应用数据增强操作到图像
            augmented_image = data_transform(image)
            # 重新命名
            file_parts = filename.split('.')
            file_parts=file_parts[0].split('_')
            file_parts[0] = file_parts[0] + '_augment'+time
            #file_parts[1] = 'png'
            name = file_parts[0]+'.png'
            print(name)
            # 构造增强图像的保存路径
            output_path = os.path.join(output_folder, name)
            # 保存增强图像
            augmented_image.save(output_path)
            print(f"Augmented image saved: {output_path}")

def copy_images(original_folder,output_folder):
    for filename in os.listdir(original_folder):
        if filename.endswith(".tif") or filename.endswith(".png") or filename.endswith(".gif"):
            # 加载原始图像
            image_path = os.path.join(original_folder, filename)
            image = Image.open(image_path)
            # 应用数据增强操作到图像
            # 重新命名
            file_parts = filename.split('.')
            file_parts[-1] = 'png'
            file1=file_parts[0].split('_')
            name =file1[0]+'_training.png'
            # 构造增强图像的保存路径
            output_path = os.path.join(output_folder, name)
            # 保存增强图像
            image.save(output_path)
            print(f"Augmented image saved: {output_path}")

if __name__ == '__main__':
    # 原始图像文件夹路径和保存增强图像的文件夹路径
    original_folder = "./DRIVE/training/masks"
    output_folder = "./data/DRIVE/background"
    #augment(original_folder=original_folder,output_folder=output_folder,time='2')
    copy_images(original_folder=original_folder,output_folder=output_folder)