import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from torchvision.models import EfficientNet_B0_Weights, ResNet50_Weights

# Grad-CAM (Digunakan pada tahap evaluasi visual)
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# ==========================================
# KONFIGURASI GLOBAL
# ==========================================
DATA_DIR = Path('./data/PlantVillage')
SUBSET_DIR = Path('./data/subset')
OUTPUT_DIR = Path('./output')

IMG_SIZE   = 224
BATCH_SIZE = 32
EPOCHS     = 20
LR         = 1e-4
SEED       = 42

SELECTED_CLASSES = [
    'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_healthy',
    'Tomato_Bacterial_spot', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Pepper__bell___Bacterial_spot'
]
NUM_CLASSES = len(SELECTED_CLASSES)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed):
    """Menetapkan seed agar eksperimen bisa direproduksi ulang (reproducible)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_subset():
    """Membuat subset dataset berdasarkan SELECTED_CLASSES secara dinamis."""
    if SUBSET_DIR.exists():
        print("[INFO] Folder subset sudah ada, melewati tahap copy dataset.")
        return

    print("[INFO] Membuat subset dataset...")
    SUBSET_DIR.mkdir(parents=True, exist_ok=True)
    
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"⚠️ Dataset asli tidak ditemukan di {DATA_DIR.absolute()}. Silakan unduh dataset PlantVillage ke dalam folder data/PlantVillage.")

    for cls in SELECTED_CLASSES:
        matches = [d for d in DATA_DIR.rglob('*') if d.is_dir() and cls.lower() in d.name.lower()]
        if matches:
            dst = SUBSET_DIR / cls
            if not dst.exists():
                shutil.copytree(str(matches[0]), str(dst))
            print(f'[OK] {cls} -> {len(list(dst.iterdir()))} gambar')
        else:
            print(f'[NOT FOUND] {cls}')


def get_dataloaders():
    """Menyiapkan dataloader dengan augmentasi gambar untuk mencegah overfitting."""
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder(root=str(SUBSET_DIR), transform=train_transform)
    
    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    # Menghapus augmentasi pada data validasi
    val_dataset.dataset = datasets.ImageFolder(root=str(SUBSET_DIR), transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, full_dataset.classes


def build_efficientnet(num_classes):
    """Membangun arsitektur EfficientNetB0 dengan head kustom."""
    model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
        
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, num_classes)
    )
    return model


def build_resnet(num_classes):
    """Membangun arsitektur ResNet50 dengan head kustom."""
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
        
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, num_classes)
    )
    return model


def train_model(model, train_loader, val_loader, epochs, lr, model_name):
    """Loop training standar PyTorch dengan Early Stopping dan LRScheduler."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    best_val_acc = 0
    patience_counter = 0
    EARLY_STOP_PATIENCE = 5
    
    OUTPUT_DIR.mkdir(exist_ok=True)
    save_path = OUTPUT_DIR / f'best_{model_name}.pth'

    for epoch in range(epochs):
        # Tahap Training
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        # Tahap Validasi
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        t_loss = train_loss / len(train_loader)
        v_loss = val_loss / len(val_loader)

        scheduler.step(v_loss)
        print(f'[{model_name}] Epoch {epoch+1}/{epochs} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f}')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f'Early stopping terpanggil di epoch {epoch+1}')
                break

    # Muat bobot model terbaik
    model.load_state_dict(torch.load(save_path))
    print(f'==> Best val accuracy {model_name}: {best_val_acc:.4f}')
    return model


def unfreeze_last_layers(model, n_layers=20):
    """Membuka (unfreeze) N layer terakhir untuk fine-tuning."""
    all_params = list(model.parameters())
    for param in all_params[-n_layers:]:
        param.requires_grad = True


def main():
    set_seed(SEED)
    print(f"🚀 [INFO] Menjalankan Pipeline di device: {DEVICE}")
    
    # Persiapan Data
    prepare_subset()
    train_loader, val_loader, class_names = get_dataloaders()
    
    # ==========================================
    # 1. Pipeline EfficientNetB0
    # ==========================================
    print('\n' + '='*50 + '\n[1] Training EfficientNetB0 (Transfer Learning)\n' + '='*50)
    model_eff = build_efficientnet(NUM_CLASSES).to(DEVICE)
    model_eff = train_model(model_eff, train_loader, val_loader, EPOCHS, LR, 'EfficientNetB0')
    
    print('\n--- Fine-Tuning EfficientNetB0 ---')
    unfreeze_last_layers(model_eff, n_layers=20)
    train_model(model_eff, train_loader, val_loader, 10, 1e-5, 'EfficientNetB0_ft')

    # ==========================================
    # 2. Pipeline ResNet50
    # ==========================================
    print('\n' + '='*50 + '\n[2] Training ResNet50 (Transfer Learning)\n' + '='*50)
    model_res = build_resnet(NUM_CLASSES).to(DEVICE)
    model_res = train_model(model_res, train_loader, val_loader, EPOCHS, LR, 'ResNet50')
    
    print('\n--- Fine-Tuning ResNet50 ---')
    unfreeze_last_layers(model_res, n_layers=20)
    train_model(model_res, train_loader, val_loader, 10, 1e-5, 'ResNet50_ft')
    
    print("\n✅ Pipeline Selesai! Model terbaik tersimpan di folder './output'")


if __name__ == '__main__':
    main()