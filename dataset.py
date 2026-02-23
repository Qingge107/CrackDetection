import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class CrackDataset(Dataset):
    def __init__(self, image_dir, mask_dir, is_train=True):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.is_train = is_train

        # 获取所有图片的文件名
        self.image_names = sorted(os.listdir(image_dir))

        # 论文要求输入图片大小为 512x512
        # 对原图进行缩放并转换为 Tensor
        self.img_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()  # 会自动将像素值从 0~255 变成 0.0~1.0
        ])

        # 对掩码(标签)图进行缩放并转换为 Tensor
        self.mask_transform = transforms.Compose([
            transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.NEAREST), # 标签图最好用最近邻插值
            transforms.ToTensor()
        ])

    def __len__(self):
        # 告诉 PyTorch 数据集里有多少张图片
        return len(self.image_names)

    def __getitem__(self, idx):
        # 1. 获取图片和掩码的文件路径
        img_name = self.image_names[idx]
        img_path = os.path.join(self.image_dir, img_name)

        # 提取去掉后缀的名字 (例如 "pic.jpg" 提取出 "pic")
        base_name = os.path.splitext(img_name)[0]

        # ================== 核心修改部分 ==================
        # 把你遇到的所有可能的掩码命名情况都列出来
        possible_mask_names = [
            f"{base_name}.png",  # 情况1：同名，后缀为png (如 pic.png)
            f"{base_name}.jpg",  # 情况1：同名，后缀为jpg (如 pic.jpg)
            f"{base_name}_mask.png",  # 情况2：加_mask，后缀为png (如 pic_mask.png)
            f"{base_name}_mask.jpg"  # 情况2：加_mask，后缀为jpg (如 pic_mask.jpg)
        ]

        mask_path = None
        # 遍历上面列出的可能性，看看文件夹里到底有哪一个
        for name in possible_mask_names:
            temp_path = os.path.join(self.mask_dir, name)
            if os.path.exists(temp_path):
                mask_path = temp_path
                break  # 找到了就立马跳出循环

        # 如果所有的可能性都找了一遍还是没有，那就只能报错了
        if mask_path is None:
            raise FileNotFoundError(f"报错啦！找不到原图 {img_name} 对应的标签图。请检查 mask 文件夹里的命名！")
        # ==================================================

        # 2. 读取图片和掩码
        image = Image.open(img_path).convert("RGB")  # 原图转为RGB彩色
        mask = Image.open(mask_path).convert("L")  # 标签图转为灰度图(单通道)

        # 3. 转换成 Tensor
        image = self.img_transform(image)
        mask = self.mask_transform(mask)

        # 4. 二值化掩码：让裂缝区域等于1，背景等于0
        # 假设掩码图中白色的裂缝，由于 ToTensor 会把 255 变成 1.0，我们这里做个阈值处理
        mask = (mask > 0.5).float()

        return image, mask