import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
from tqdm import tqdm  # 引入进度条工具，批量处理时看着更爽

# 从你的 model.py 中导入模型结构
from model import BiCrack


# ==========================================
# 功能 1：预测单张图片并弹窗对比
# ==========================================
def predict_single_image(image_path, weight_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"正在使用 {device} 进行单张推理...")

    model = BiCrack(num_classes=1).to(device)
    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path, map_location=device))
        print("✅ 成功加载模型权重！")
    else:
        print(f"❌ 找不到权重文件：{weight_path}")
        return
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])

    img = Image.open(image_path).convert('RGB')
    original_size = img.size
    img_tensor = transform(img).unsqueeze(0).to(device)

    print("🧠 AI 正在思考中...")
    with torch.no_grad():
        output = model(img_tensor)
        prob = torch.sigmoid(output)
        mask = (prob > 0.5).float()

    mask = mask.squeeze().cpu().numpy()
    mask_image = Image.fromarray((mask * 255).astype('uint8'))
    # 注意这里已经改成了 Image.Resampling.NEAREST 修复了之前的报错
    mask_image = mask_image.resize(original_size, Image.Resampling.NEAREST)

    save_path = "result_single_crack.png"
    mask_image.save(save_path)
    print(f"🎉 预测完成！结果已保存为 {save_path}")

    # 画图对比
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(img)
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.title("AI Predicted Crack")
    plt.imshow(mask_image, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


# ==========================================
# 功能 2：批量预测整个文件夹（新增功能！）
# ==========================================
def predict_folder(input_folder, output_folder, weight_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"正在使用 {device} 进行批量推理...")

    # 1. 加载模型（批量预测时，模型只需要加载一次！）
    model = BiCrack(num_classes=1).to(device)
    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path, map_location=device))
        print("✅ 成功加载模型权重！")
    else:
        print(f"❌ 找不到权重文件：{weight_path}")
        return
    model.eval()

    # 2. 如果输出文件夹不存在，帮用户自动创建一个
    os.makedirs(output_folder, exist_ok=True)

    # 3. 图像预处理工具
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])

    # 4. 获取输入文件夹里所有的图片
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    # 过滤掉非图片文件
    image_names = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_extensions)]

    if len(image_names) == 0:
        print(f"⚠️ 哎呀，在 {input_folder} 文件夹里没有找到任何图片！")
        return

    print(f"📂 共找到 {len(image_names)} 张图片，开始流水线批量预测...")

    # 5. 开始批量循环预测，并加上进度条
    for img_name in tqdm(image_names, desc="批量预测进度"):
        img_path = os.path.join(input_folder, img_name)

        # 读取图片
        img = Image.open(img_path).convert('RGB')
        original_size = img.size
        img_tensor = transform(img).unsqueeze(0).to(device)

        # 前向传播
        with torch.no_grad():
            output = model(img_tensor)
            prob = torch.sigmoid(output)
            mask = (prob > 0.5).float()

        # 转回图片
        mask = mask.squeeze().cpu().numpy()
        mask_image = Image.fromarray((mask * 255).astype('uint8'))
        mask_image = mask_image.resize(original_size, Image.Resampling.NEAREST)

        # 保存结果：名字和原图一样，存在输出文件夹里
        save_path = os.path.join(output_folder, img_name)
        mask_image.save(save_path)

    print(f"\n🎉 批量预测大功告成！所有的黑白掩码图都存在这里啦：{output_folder}")


if __name__ == "__main__":
    # ================= 使用控制台 =================

    # 你的权重文件路径
    WEIGHT_PATH = "weights/bicrack_epoch_80.pth"

    # 【模式选择】
    # 如果想测单张图片，把 MODE 设置为 1
    # 如果想测整个文件夹，把 MODE 设置为 2
    MODE = 1

    if MODE == 1:
        # 单张图片预测的路径配置
        TEST_IMAGE = "20260221_202953.jpg"  # <-- 换成你的单张图路径
        predict_single_image(TEST_IMAGE, WEIGHT_PATH)

    elif MODE == 2:
        # 批量预测的路径配置
        INPUT_FOLDER = "test_images"  # <-- 把你要测试的图片全丢进这个新建的文件夹里
        OUTPUT_FOLDER = "test_results"  # <-- AI 画出来的裂缝图会自动存进这里面

        # 运行批量预测
        predict_folder(INPUT_FOLDER, OUTPUT_FOLDER, WEIGHT_PATH)