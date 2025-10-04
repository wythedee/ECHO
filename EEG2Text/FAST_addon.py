import torch
import torch.nn.functional as F
import einops
from torch import nn

class Conv2d_Mimic(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding

        # Initialize weights using nn.Linear
        self.unfold = nn.Unfold(kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        self.linear = nn.Linear(in_channels * self.kernel_size[0] * self.kernel_size[1], out_channels, bias=bias)

    def forward(self, x):
        # Unfold input to get sliding windows, shape: [batch_size, in_channels * kernel_h * kernel_w, L]
        x_unfolded = self.unfold(x)

        # Perform convolution using linear layer, shape: [batch_size, L, in_channels * kernel_h * kernel_w]
        x_unfolded = x_unfolded.permute(0, 2, 1)
        out = self.linear(x_unfolded)  # shape: [batch_size, L, out_channels]

        # Reshape to final output shape
        out_h = (x.shape[2] + 2*self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        out_w = (x.shape[3] + 2*self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1
        out = out.permute(0, 2, 1).contiguous().view(x.shape[0], self.out_channels, out_h, out_w)

        return out
    
class AD1(nn.Module):
    def __init__(self, channels, dim=32):
        super().__init__()
        self.cnn1 = nn.Conv2d(1, dim, (1, 5), bias=True)
        self.cnn2 = Conv2d_Mimic(dim, dim, (channels, 1), padding=0, bias=False)
        self.cnn3 = nn.Conv2d(dim, dim, (1, 5), padding=(0, 2), bias=False)
        self.cnn4 = nn.Conv2d(dim, dim, (1, 5), padding=(0, 2), bias=False)

    def forward(self, x):
        x = einops.rearrange(x, 'B C T -> B 1 C T')
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = self.cnn4(x)
        x = F.gelu(x)
        x = einops.reduce(x, 'B F 1 T -> B F', 'mean')
        return x
    
class AD2(nn.Module):
    def __init__(self, channels, dim=32):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, dim, (1, 7), padding=(0, 3), bias=True),
            Conv2d_Mimic(dim, dim, (channels, 1), padding=0, bias=False),
            nn.GELU(),
            nn.MaxPool2d((1, 5), stride=(1, 2), padding=(0, 1)),
            nn.Conv2d(dim, dim, (1, 5), padding=(0, 2), bias=False),
            nn.Conv2d(dim, dim, (1, 5), padding=(0, 2), bias=False),
            nn.GELU(),
        )

    def forward(self, x):
        x = einops.rearrange(x, 'B C T -> B 1 C T')
        x = self.cnn(x)
        x = einops.reduce(x, 'B F 1 T -> B F', 'mean')
        return x

class Conv2d_AE(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding

        # Initialize weights using nn.Linear
        self.unfold = nn.Unfold(kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        # self.linear = nn.Linear(in_channels * self.kernel_size[0] * self.kernel_size[1], out_channels, bias=bias)
        self.linear = nn.Sequential(
            nn.Linear(in_channels * self.kernel_size[0] * self.kernel_size[1], out_channels//2, bias=bias),
            nn.GELU(),
            nn.Linear(out_channels//2, out_channels, bias=bias)
        )

    def forward(self, x):
        # Unfold input to get sliding windows, shape: [batch_size, in_channels * kernel_h * kernel_w, L]
        x_unfolded = self.unfold(x)

        # Perform convolution using linear layer, shape: [batch_size, L, in_channels * kernel_h * kernel_w]
        x_unfolded = x_unfolded.permute(0, 2, 1)
        out = self.linear(x_unfolded)  # shape: [batch_size, L, out_channels]

        # Reshape to final output shape
        out_h = (x.shape[2] + 2*self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        out_w = (x.shape[3] + 2*self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1
        out = out.permute(0, 2, 1).contiguous().view(x.shape[0], self.out_channels, out_h, out_w)

        return out

class AD3(nn.Module):
    def __init__(self, channels, dim=32):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, dim, (1, 7), padding=(0, 3), bias=True),
            Conv2d_AE(dim, dim, (channels, 1), padding=0, bias=False),
            nn.GELU(),
            nn.MaxPool2d((1, 5), stride=(1, 2), padding=(0, 2)),
            nn.Conv2d(dim, dim, (1, 5), padding=(0, 2), bias=False),
            nn.Conv2d(dim, dim, (1, 5), padding=(0, 2), bias=False),
            nn.GELU(),
        )

    def forward(self, x):
        x = einops.rearrange(x, 'B C T -> B 1 C T')
        x = self.cnn(x)
        x = einops.reduce(x, 'B F 1 T -> B F', 'mean')
        return x

if __name__ == "__main__":
    ad1 = AD3(channels=10)
    input_tensor = torch.randn(8, 10, 100)  # Batch size of 8, 64 channels, sequence length of 100
    output_tensor = ad1(input_tensor)
    print(output_tensor.shape)  # Expected output shape: [8, dim]