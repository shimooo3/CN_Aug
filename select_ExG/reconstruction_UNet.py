
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import argparse
from tqdm import tqdm
from pathlib import Path

# train_controlnet_v71.pyからReconstructionDecoderクラスを定義
class ReconstructionDecoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super().__init__()
        # Input is (N, in_channels, H/8, W/8) for resolution 512
        # Output should be (N, out_channels, H, W)
        
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Upsample(scale_factor=2, mode='nearest'), # H/4
            torch.nn.Conv2d(256, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Upsample(scale_factor=2, mode='nearest'), # H/2
            torch.nn.Conv2d(128, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Upsample(scale_factor=2, mode='nearest'), # H
            torch.nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)

# 実画像をエンコードするエンコーダ
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            # H -> H/2
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # H/2 -> H/4
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # H/4 -> H/8
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # H/8 -> H/8
            nn.Conv2d(256, 320, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.layers(x)

# U-Net (エンコーダ-デコーダ) モデル
# NOTE: このモデルは厳密なU-Netの定義（スキップ接続を持つ）とは異なりますが、
#       ユーザーの要求に基づき、エンコーダとデコーダを組み合わせた構造です。
class CompositionUNet(nn.Module):
    def __init__(self, pretrained_decoder_path=None, freeze_decoder=True):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = ReconstructionDecoder(in_channels=320, out_channels=1)
        
        if pretrained_decoder_path:
            print(f"Loading pretrained decoder weights from {pretrained_decoder_path}")
            self.decoder.load_state_dict(torch.load(pretrained_decoder_path))
        
        if freeze_decoder:
            print("Freezing decoder weights.")
            for param in self.decoder.parameters():
                param.requires_grad = False

    def forward(self, x):
        features = self.encoder(x)
        output = self.decoder(features)
        return output

# データセットクラス
class ImagePairDataset(Dataset):
    def __init__(self, real_dir, comp_dir, transform=None, comp_transform=None):
        self.real_dir = real_dir
        self.comp_dir = comp_dir
        self.transform = transform
        self.comp_transform = comp_transform
        
        self.real_images = sorted([f for f in os.listdir(real_dir) if os.path.isfile(os.path.join(real_dir, f))])
        self.comp_images = sorted([f for f in os.listdir(comp_dir) if os.path.isfile(os.path.join(comp_dir, f))])
        
        if len(self.real_images) != len(self.comp_images):
            raise ValueError("The number of images in real_dir and comp_dir must be the same.")

    def __len__(self):
        return len(self.real_images)

    def __getitem__(self, idx):
        real_path = os.path.join(self.real_dir, self.real_images[idx])
        comp_path = os.path.join(self.comp_dir, self.comp_images[idx])
        
        real_image = Image.open(real_path).convert("RGB")
        comp_image = Image.open(comp_path).convert("L") # グレースケール

        if self.transform:
            real_image = self.transform(real_image)
        if self.comp_transform:
            comp_image = self.comp_transform(comp_image)
            
        return real_image, comp_image

def masked_mse_loss(pred, target, white_only=True):
    """
    損失関数を計算する。
    white_only=Trueの場合、ターゲット画像の白い領域と、その8近傍の黒い領域のみで損失を計算する。
    これにより、輪郭部分の学習を促進し、予測が真っ黒になるのを防ぐ。
    それ以外の場合は、画像全体で損失を計算する。
    """
    if white_only:
        # 白い領域のマスク (ピクセル値 > 0.5)
        white_mask = (target > 0.5).float()
        
        # 白い領域の8近傍を計算 (最大プーリングで膨張させる)
        # カーネルサイズ3x3で最大値を取ることで、白い領域が1ピクセル広がる
        dilated_white_mask = F.max_pool2d(white_mask, kernel_size=3, stride=1, padding=1)
        
        # 膨張した領域から元の白い領域を引くことで、境界領域のみのマスクを得る
        boundary_mask = dilated_white_mask - white_mask
        
        # 境界領域のうち、実際に黒い部分のみを抽出する
        black_boundary_mask = boundary_mask * (target <= 0.5).float()
        
        # 最終的なマスクは「白い領域」+「境界の黒い領域」
        final_mask = white_mask + black_boundary_mask
        
        # マスクを適用して損失を計算
        loss = F.mse_loss(pred, target, reduction="none")
        loss = (loss * final_mask).sum() / (final_mask.sum() + 1e-8)
    else:
        loss = F.mse_loss(pred, target)
    return loss

def save_validation_images(model, dataloader, device, save_dir, epoch):
    model.eval()
    with torch.no_grad():
        real_imgs, target_imgs = next(iter(dataloader))
        real_imgs = real_imgs.to(device)
        
        output_imgs = model(real_imgs)
        
        # 画像を保存
        os.makedirs(save_dir, exist_ok=True)
        grid = []
        to_pil = transforms.ToPILImage()
        
        num_images = min(real_imgs.size(0), 4)
        for i in range(num_images):
            grid.append(to_pil(real_imgs[i].cpu()))
            grid.append(to_pil(target_imgs[i].cpu()))
            grid.append(to_pil(output_imgs[i].cpu()))
            
        # 3列のグリッド画像を作成
        w, h = grid[0].size
        result_img = Image.new(grid[0].mode, (w * 3, h * num_images))
        for i, img in enumerate(grid):
            result_img.paste(img, box=(i % 3 * w, i // 3 * h))
            
        result_img.save(os.path.join(save_dir, f"validation_epoch_{epoch}.png"))

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) # for real images
    ])
    comp_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])

    # Dataset and DataLoader
    dataset = ImagePairDataset(args.real_image_dir, args.composition_image_dir, transform, comp_transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    val_dataloader = DataLoader(dataset, batch_size=4, shuffle=False)


    # Model
    model = CompositionUNet(
        pretrained_decoder_path=args.decoder_weights_path,
        freeze_decoder=True # Always start with decoder frozen
    ).to(device)

    # Optimizer
    print("Initially training only the encoder.")
    optimizer = torch.optim.Adam(model.encoder.parameters(), lr=args.lr)

    # Output directories
    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    validation_dir = output_dir / "validation"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    validation_dir.mkdir(parents=True, exist_ok=True)

    # Checkpoint tracking
    best_loss = float('inf')
    saved_checkpoints = []
    saved_decoder_checkpoints = []

    # Training loop
    decoder_unfrozen = False
    for epoch in range(args.epochs):
        # Unfreeze decoder if the epoch is reached
        if not decoder_unfrozen and args.unfreeze_decoder_from_epoch != -1 and epoch >= args.unfreeze_decoder_from_epoch:
            print(f"Epoch {epoch+1}: Unfreezing decoder weights for fine-tuning.")
            for param in model.decoder.parameters():
                param.requires_grad = True
            
            print("Creating new optimizer for the whole model.")
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            decoder_unfrozen = True

        model.train()
        
        use_white_only_loss = True
        if args.use_all_region_from_epoch != -1 and epoch >= args.use_all_region_from_epoch:
            use_white_only_loss = False
            if epoch == args.use_all_region_from_epoch:
                print(f"Epoch {epoch+1}: Switched to using all regions for loss calculation.")

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        total_loss = 0

        for real_images, comp_images in progress_bar:
            real_images = real_images.to(device)
            comp_images = comp_images.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(real_images)
            
            # Loss calculation
            loss = masked_mse_loss(outputs, comp_images, white_only=use_white_only_loss)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{args.epochs}], Average Loss: {avg_loss:.4f}")

        # Save validation images (behavior is unchanged)
        if (epoch + 1) % args.validation_interval == 0:
            save_validation_images(model, val_dataloader, device, validation_dir, epoch + 1)
            print(f"Saved validation images for epoch {epoch+1}")

        # Save checkpoint only if loss improves, and keep only the best 3
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint_path = checkpoint_dir / f"model_epoch_{epoch+1}_loss_{avg_loss:.4f}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Loss improved. Saved checkpoint to {checkpoint_path}")
            
            saved_checkpoints.append(checkpoint_path)
            
            if len(saved_checkpoints) > 3:
                checkpoint_to_remove = saved_checkpoints.pop(0)
                try:
                    os.remove(checkpoint_to_remove)
                    print(f"Removed old checkpoint: {checkpoint_to_remove}")
                except OSError as e:
                    print(f"Error removing old checkpoint {checkpoint_to_remove}: {e}")

            if decoder_unfrozen:
                decoder_checkpoint_path = checkpoint_dir / f"decoder_epoch_{epoch+1}_loss_{avg_loss:.4f}.pth"
                torch.save(model.decoder.state_dict(), decoder_checkpoint_path)
                print(f"Saved decoder checkpoint to {decoder_checkpoint_path}")
                
                saved_decoder_checkpoints.append(decoder_checkpoint_path)
                
                if len(saved_decoder_checkpoints) > 3:
                    decoder_checkpoint_to_remove = saved_decoder_checkpoints.pop(0)
                    try:
                        os.remove(decoder_checkpoint_to_remove)
                        print(f"Removed old decoder checkpoint: {decoder_checkpoint_to_remove}")
                    except OSError as e:
                        print(f"Error removing old decoder checkpoint {decoder_checkpoint_to_remove}: {e}")

    # Save final model
    final_model_path = output_dir / "final_model.pth"
    torch.save(model.state_dict(), final_model_path)
    print(f"Training complete. Final model saved to {final_model_path}")


def main():
    parser = argparse.ArgumentParser(description="Train a U-Net to convert real images to composition images.")
    parser.add_argument("--real_image_dir", type=str, required=True, help="Directory with real images.")
    parser.add_argument("--composition_image_dir", type=str, required=True, help="Directory with composition images.")
    parser.add_argument("--decoder_weights_path", type=str, required=True, help="Path to the pretrained reconstruction decoder weights (.pt file).")
    parser.add_argument("--output_dir", type=str, default="reconU_output", help="Directory to save checkpoints and results.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--image_size", type=int, default=512, help="Size to resize images to.")
    parser.add_argument("--unfreeze_decoder_from_epoch", type=int, default=-1, help="Epoch to start unfreezing and training the decoder. Default: -1 (decoder remains frozen).")
    parser.add_argument("--validation_interval", type=int, default=50, help="Run validation every N epochs.")
    parser.add_argument("--use_all_region_from_epoch", type=int, default=-1, help="Epoch from which to start using all regions for loss calculation. Default: -1 (only white regions).")

    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    main()
