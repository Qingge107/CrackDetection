import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os
from model import BiCrack
from loss import BiCrackLoss
from dataset import CrackDataset


# 计算评价指标
def calculate_metrics(preds, targets):
    preds = (torch.sigmoid(preds) > 0.5).float()
    tp = (preds * targets).sum().item()
    fp = (preds * (1 - targets)).sum().item()
    fn = ((1 - preds) * targets).sum().item()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    return precision, recall, f1


def main():
    # ================= 1. 基础设置 =================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"当前使用设备: {device}")

    EPOCHS = 150
    BATCH_SIZE = 8
    LR = 0.0001
    image_dir = "dataset/images"
    mask_dir = "dataset/masks"

    if not os.path.exists(image_dir) or len(os.listdir(image_dir)) == 0:
        print("错误：请检查 dataset/images 文件夹是否为空！")
        return

    # ================= 2. 加载数据集 =================
    print("正在加载数据集...")
    full_dataset = CrackDataset(image_dir, mask_dir)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"总图片数: {len(full_dataset)} | 训练集: {train_size} | 测试集: {val_size}")

    # ================= 3. 初始化模型与优化器 =================
    model = BiCrack(num_classes=1).to(device)
    criterion = BiCrackLoss(w_bce=0.5, w_dice=0.5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.cuda.amp.GradScaler()

    best_f1 = 0.0
    save_dir = "weights"
    os.makedirs(save_dir, exist_ok=True)  # 创建保存权重的文件夹

    # ================= 4. 开始训练循环 =================
    for epoch in range(EPOCHS):
        print(f"\nEpoch [{epoch + 1}/{EPOCHS}]")

        # --- 训练阶段 ---
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc="Training")

        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)

        # --- 验证阶段 ---
        model.eval()
        val_loss, val_precision, val_recall, val_f1 = 0, 0, 0, 0

        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc="Validating"):
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

                p, r, f1 = calculate_metrics(outputs, masks)
                val_precision += p
                val_recall += r
                val_f1 += f1

        # 计算平均指标
        avg_val_loss = val_loss / len(val_loader)
        avg_p = val_precision / len(val_loader)
        avg_r = val_recall / len(val_loader)
        avg_f1 = val_f1 / len(val_loader)

        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"Val Precision: {avg_p:.4f} | Val Recall: {avg_r:.4f} | Val F1: {avg_f1:.4f}")

        # ================= 5. 保存模型权重 =================
        # 5.1 保存当前epoch的权重（新增部分）
        current_save_path = os.path.join(save_dir, f"bicrack_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), current_save_path)
        print(f"💾 已保存当前epoch模型至 {current_save_path}")

        # 5.2 保存最佳模型（原有逻辑）
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            best_save_path = os.path.join(save_dir, "bicrack_best.pth")
            torch.save(model.state_dict(), best_save_path)
            print(f"⭐ 发现更好的模型！已保存至 {best_save_path} (F1: {best_f1:.4f})")


if __name__ == "__main__":
    main()