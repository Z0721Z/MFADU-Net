import torch
from torchvision import transforms
from PIL import Image

# 1. 读取图像
image_path = "ISBI/images/ISIC_0000007.png" # 替换为实际的图像路径
image = Image.open(image_path)

# 2. 定义图像转换操作
transform = transforms.Compose([
    transforms.ToTensor(), # 将图像转换为张量

])

# 3. 应用图像转换
tensor_image = transform(image)
# 打印张量的形状
print("Tensor shape:", tensor_image.shape)