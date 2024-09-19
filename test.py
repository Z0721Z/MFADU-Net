import torch
from torchvision import transforms
from PIL import Image

image_path = "ISBI/images/ISIC_0000007.png" 
image = Image.open(image_path)

transform = transforms.Compose([
    transforms.ToTensor(),
])

tensor_image = transform(image)
print("Tensor shape:", tensor_image.shape)
