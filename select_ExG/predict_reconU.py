import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import numpy as np

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


def predict(model, image_path, output_path, image_size=512, threshold=0.5):
    """
    Loads an image, runs it through the model, binarizes the output,
    applies a CIVE-based filter, and saves it.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform_for_unet = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Input image not found at {image_path}")
        return

    resized_image = image.resize((image_size, image_size), Image.BILINEAR)

    image_np = np.array(resized_image)
    r, g, b = image_np[:, :, 0], image_np[:, :, 1], image_np[:, :, 2]
    cive = 0.441 * r - 0.811 * g + 0.385 * b + 18.78745
    cive_mask = cive < 0

    input_tensor = transform_for_unet(resized_image).unsqueeze(0).to(device)

    model.to(device)
    model.eval()
    with torch.no_grad():
        output_tensor = model(input_tensor)

    unet_output_tensor = output_tensor.squeeze(0).cpu()
    unet_mask_tensor = (unet_output_tensor > threshold).float()
    unet_mask_np = unet_mask_tensor.squeeze(0).numpy() # 1.0 for white, 0.0 for black
    unet_image = Image.fromarray((unet_mask_np * 255).astype(np.uint8), mode='L')

    cive_image_np = np.where(cive_mask, 0, 1)
    cive_image = Image.fromarray((cive_image_np * 255).astype(np.uint8), mode='L')

    final_mask_np = unet_mask_np.copy()
    condition = (final_mask_np == 1) & (cive_mask) # U-Net is white AND CIVE is positive
    final_mask_np[condition] = 0 # Set to black
    combined_image = Image.fromarray((final_mask_np * 255).astype(np.uint8), mode='L')

    images = [cive_image, unet_image, combined_image]
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    composite_image = Image.new('L', (total_width, max_height))

    x_offset = 0
    for im in images:
        composite_image.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    # Save the composite image
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    composite_image.save(output_path)
    print(f"Composite image saved to {output_path}")
    print("From left to right: CIVE binarization (cive>0 is black), U-Net binarization, Combined result (AND)")


if __name__ == "__main__":
    MODEL_WEIGHTS_PATH = "./09_train_UNet/output/checkpoints/model_epoch_2904_loss_0.0433.pth"
    INPUT_IMAGE_PATH = "./09_train_UNet/train/target/NON0333_960x1280_2010.jpg"
    INPUT_IMAGE_PATH = "./09_train_UNet/NON0335_960x1280_0410.jpg"
    OUTPUT_IMAGE_PATH = "./09_train_UNet/bina_reconU_cive.png" 

    if not os.path.exists(MODEL_WEIGHTS_PATH):
        print(f"Error: Model weights not found at {MODEL_WEIGHTS_PATH}")
        print("Please train the model using reconU.py first or update the path.")
    else:
        model = CompositionUNet()

        print(f"Loading model weights from {MODEL_WEIGHTS_PATH}")
        model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH))

        predict(model, INPUT_IMAGE_PATH, OUTPUT_IMAGE_PATH)
