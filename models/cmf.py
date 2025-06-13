import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage.feature import local_binary_pattern
from skimage.color import rgb2gray


class CMFBlock(nn.Module):
    def __init__(self, input_channels, output_channels=1, dropout_prob=0.1, num_heads=8):
        super(CMFBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.gelu1 = nn.GELU()
        self.bn1 = nn.BatchNorm2d(64)
        self.dropout1 = nn.Dropout(dropout_prob)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.gelu2 = nn.GELU()
        self.bn2 = nn.BatchNorm2d(128)
        self.dropout2 = nn.Dropout(dropout_prob)

        self.attn = nn.MultiheadAttention(embed_dim=128, num_heads=num_heads, dropout=dropout_prob)

        self.conv3 = nn.Conv2d(128, output_channels, kernel_size=3, padding=1)
        self.gelu3 = nn.GELU()
        self.bn3 = nn.BatchNorm2d(output_channels)
        self.dropout3 = nn.Dropout(dropout_prob)

    def compute_spectrum(self, img_tensor):
        if img_tensor.ndimension() == 4 and img_tensor.shape[1] == 3:
            img_tensor = img_tensor.mean(dim=1, keepdim=True)

        img_tensor = img_tensor.float()

        f = torch.fft.fft2(img_tensor)
        
        fshift = torch.fft.fftshift(f)

        magnitude_spectrum = torch.abs(fshift)
        magnitude_spectrum = torch.log(magnitude_spectrum + 1)

        magnitude_spectrum = magnitude_spectrum / magnitude_spectrum.max() * 255

        magnitude_spectrum = magnitude_spectrum.to(torch.float32)

        magnitude_spectrum = F.interpolate(magnitude_spectrum, size=(224, 224), mode='bilinear', align_corners=False)
        return magnitude_spectrum
    
    def batch_lbp_processing(self, input_tensor, radius=1, n_points=8, method='uniform'):
        device = input_tensor.device
        
        input_numpy = input_tensor.cpu().numpy()
        
        lbp_images = []
        
        for i in range(input_numpy.shape[0]):
            image = input_numpy[i]
            
            gray_image = rgb2gray(image.transpose(1, 2, 0))
            
            lbp_result = np.zeros_like(gray_image, dtype=np.uint8)
            
            for channel in range(3):
                lbp_channel = local_binary_pattern(image[channel], n_points, radius, method)
                lbp_result += lbp_channel.astype(np.uint8)
            
            lbp_images.append(lbp_result)

        lbp_images = np.array(lbp_images)

        lbp_tensor = torch.tensor(lbp_images, dtype=torch.float32).to(device)
        
        lbp_tensor = lbp_tensor.unsqueeze(1)
        
        return lbp_tensor

    def forward(self, rgb_input, raw):
        spectrum = self.compute_spectrum(raw)
        ldp = self.batch_lbp_processing(raw)

        output = torch.cat([rgb_input, spectrum, ldp], dim=1)

        output = self.conv1(output)
        output = self.gelu1(output)
        output = self.bn1(output)
        output = self.dropout1(output)

        output = self.conv2(output)
        output = self.gelu2(output)
        output = self.bn2(output)
        output = self.dropout2(output)

        batch_size, channels, height, width = output.shape
        output_flat = output.flatten(2).transpose(1, 2)
        
        attn_output, _ = self.attn(output_flat, output_flat, output_flat)
        
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, channels, height, width)
        
        output = attn_output + output
        
        output = self.conv3(output)
        output = self.gelu3(output)
        output = self.bn3(output)
        output = self.dropout3(output)

        return output
