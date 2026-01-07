import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
from typing import List, Tuple
warnings.filterwarnings("ignore")   # do not show unnecessary figure axis warnings

def create_dog_filter(sigma1=1.0, sigma2=1.2, size=3, device='cpu'):
    """Create a Difference of Gaussian (DoG) filter.
    
    Args:
        sigma1: Standard deviation of the first Gaussian (smaller) - increased to 1.0 for weaker edge detection
        sigma2: Standard deviation of the second Gaussian (larger) - kept close to sigma1 for weaker contrast
        size: Size of the filter kernel - reduced to 3 for smaller effect
        device: Device to create tensor on ('cpu' or 'cuda')
    
    Returns:
        DoG filter tensor
    """
    # Create coordinate grid
    x = torch.arange(-(size//2), size//2 + 1, dtype=torch.float32, device=device)
    y = torch.arange(-(size//2), size//2 + 1, dtype=torch.float32, device=device)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    # Calculate distances from center
    r2 = X**2 + Y**2
    
    # Create two Gaussian kernels with reduced difference
    g1 = torch.exp(-r2 / (2 * sigma1**2)) / (2 * np.pi * sigma1**2)
    g2 = torch.exp(-r2 / (2 * sigma2**2)) / (2 * np.pi * sigma2**2)
    
    # Create DoG filter with reduced contrast
    dog = g1 - 0.6 * g2  # Further reduced contrast by multiplying g2 by 0.6
    
    # Normalize the filter
    dog = dog / torch.sum(torch.abs(dog))
    
    return dog.unsqueeze(0).unsqueeze(0)  # Add channel dimensions [1, 1, size, size]

def apply_dog_filter(image, dog_filter):
    """Apply DoG filter to an image with multiple color channels.
    
    Args:
        image: Input image tensor [batch_size, channels, height, width]
        dog_filter: DoG filter tensor [1, 1, size, size]
    
    Returns:
        Filtered image tensor with multiple channels:
        - Luminance channel (R+G+B)/3
        - R-G opponency channel
        - B-G opponency channel
    """
    batch_size = image.size(0)
    
    # Extract RGB channels
    R = image[:, 0:1]  # Red channel
    G = image[:, 1:2]  # Green channel
    B = image[:, 2:3]  # Blue channel
    
    # Calculate luminance (R+G+B)/3
    luminance = (R + G + B) / 3.0
    
    # Calculate color opponency channels
    R_G = R - G  # Red-Green opponency
    B_G = B - G  # Blue-Green opponency
    
    # Apply DoG filter to each channel
    luminance_filtered = F.conv2d(luminance, dog_filter, padding='same')
    R_G_filtered = F.conv2d(R_G, dog_filter, padding='same')
    B_G_filtered = F.conv2d(B_G, dog_filter, padding='same')
    
    # Normalize each channel
    luminance_filtered = (luminance_filtered - luminance_filtered.min()) / (luminance_filtered.max() - luminance_filtered.min())
    R_G_filtered = (R_G_filtered - R_G_filtered.min()) / (R_G_filtered.max() - R_G_filtered.min())
    B_G_filtered = (B_G_filtered - B_G_filtered.min()) / (B_G_filtered.max() - B_G_filtered.min())
    
    # Concatenate all channels
    filtered = torch.cat([luminance_filtered, R_G_filtered, B_G_filtered], dim=1)
    
    return filtered

def create_gabor_filters(frequencies: List[float] = [0.25, 0.5, 1, 2],  # Further increased frequencies for more detailed features
                        orientations: List[float] = [0, 45, 90, 135],
                        phases: List[float] = [0, np.pi/2],  # 0 and π/2 for both polarities
                        device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor]:
    """Create a bank of Gabor filters with specified parameters following Rolls' 2021 implementation.
    
    Args:
        frequencies: List of frequencies for the Gabor filters (default: [0.5, 1, 2, 4])
                    - 0.5 cycle/image: captures medium features (64 pixels/cycle)
                    - 1 cycle/image: captures small features (32 pixels/cycle)
                    - 2 cycle/image: captures very small features (16 pixels/cycle)
                    - 4 cycle/image: captures extremely small features (8 pixels/cycle)
        orientations: List of orientations in degrees (default: [0, 45, 90, 135])
        phases: List of phases in radians (default: [0, π/2] for both polarities)
        device: Device to create tensors on ('cpu' or 'cuda')
    
    Returns:
        Tuple of (real_filters, imag_filters) tensors with shape [out_channels, in_channels, height, width]
    """
    
    real_filters = []
    imag_filters = []
    
    # Convert orientations from degrees to radians
    orientations_rad = [np.pi * angle / 180 for angle in orientations]
    
    # Constants from Rolls' code
    dd = torch.tensor(1.0 / np.sqrt(2.0 * np.pi), dtype=torch.float32, device=device)
    N_ANGLES = len(orientations_rad)
    TH0 = torch.tensor(np.pi / N_ANGLES * 2.0, dtype=torch.float32, device=device)  # angle between filters
    
    # Define filter sizes for each frequency (based on wavelength)
    # Size = 2 * wavelength to ensure proper sampling
    filter_sizes = [7, 7, 7, 7]  # Sizes corresponding to frequencies [0.5, 1, 2, 4]
    



    frequencies = [1/3.5, 2/3.5, 4/3.5, 8/3.5]
    filter_sizes = [4, 8, 16, 32]
    

    
    for idx, freq in enumerate(frequencies):
        # Get specific filter size
        gsize = filter_sizes[idx]
        
        # Calculate frequency scaling
        auxgaba = 2 ** -(freq - 0.25)  # Adjusted frequency scaling for lower frequencies
        auxgabaa = 2 ** -(freq - 0.25)
        
        print(f"Creating filter for frequency {freq}:")
        print(f"- Filter size: {gsize}x{gsize} pixels")
        print(f"- Wavelength: {32/freq:.1f} pixels/cycle")
        print(f"- Frequency scaling: {auxgaba:.4f}")
        
        for ang in range(N_ANGLES):
            for phase in phases:
                # Calculate rotation parameters
                auxgabc = torch.tensor(np.cos(ang * TH0.item()), dtype=torch.float32, device=device)
                auxgabs = torch.tensor(np.sin(ang * TH0.item()), dtype=torch.float32, device=device)
                
                # Create coordinate grids
                i = torch.arange(1, gsize + 1, dtype=torch.float32, device=device)
                j = torch.arange(1, gsize + 1, dtype=torch.float32, device=device)
                I, J = torch.meshgrid(i, j, indexing='ij')
                
                # Calculate rotated coordinates
                temp_x = auxgaba * I - auxgabaa * gsize/2
                temp_y = auxgaba * J - auxgabaa * gsize/2
                x = temp_x * auxgabc + temp_y * auxgabs
                y = temp_y * auxgabc - temp_x * auxgabs
                
                # Create Gabor filter with phase
                temp = dd * auxgaba * torch.exp(-(4 * x * x + y * y) / 8)
                pi_tensor = torch.tensor(np.pi, dtype=torch.float32, device=device)
                phase_tensor = torch.tensor(phase, dtype=torch.float32, device=device)
                
                # Real part (cosine)
                gabor_r = temp * torch.cos(pi_tensor * x + phase_tensor)
                # Imaginary part (sine)
                gabor_i = temp * torch.sin(pi_tensor * x + phase_tensor)
                
                # Normalize filters
                gabor_r = gabor_r / torch.sum(torch.abs(gabor_r))
                gabor_i = gabor_i / torch.sum(torch.abs(gabor_i))
                
                # Create 32x32 tensors filled with zeros
                gabor_r_padded = torch.zeros((32, 32), dtype=torch.float32, device=device)
                gabor_i_padded = torch.zeros((32, 32), dtype=torch.float32, device=device)
                
                # Calculate padding
                pad_size = (32 - gsize) // 2
                
                # Place the filter in the center of the padded tensor
                gabor_r_padded[pad_size:pad_size+gsize, pad_size:pad_size+gsize] = gabor_r
                gabor_i_padded[pad_size:pad_size+gsize, pad_size:pad_size+gsize] = gabor_i
                
                real_filters.append(gabor_r_padded)
                imag_filters.append(gabor_i_padded)
    
    # Stack filters and add channel dimension [out_channels, in_channels, height, width]
    real_filters = torch.stack(real_filters).unsqueeze(1)  # Add input channel dimension
    imag_filters = torch.stack(imag_filters).unsqueeze(1)  # Add input channel dimension
    
    print(f"\nCreated Gabor filter bank:")
    print(f"- {len(frequencies)} frequencies: {frequencies}")
    print(f"- {len(orientations)} orientations: {orientations}°")
    print(f"- {len(phases)} phases: {[p*180/np.pi for p in phases]}°")
    print(f"Total filters: {len(frequencies) * len(orientations) * len(phases)} = {len(frequencies) * len(orientations) * len(phases)}")
    print(f"Filter shape: {real_filters.shape}")
    
    return real_filters, imag_filters

class L1Layer(nn.Module):
    def __init__(self, input_shape, output_size, device='cpu', learning_rate=0.000001, receptive_field_radius=3):
        super(L1Layer, self).__init__()
        self.device = device
        self.learning_rate = learning_rate
        self.receptive_field_radius = receptive_field_radius
        
        # Calculate input dimensions
        self.input_height, self.input_width = input_shape[1], input_shape[2]  # 32x32
        self.num_channels = input_shape[0]  # Number of input channels (Gabor filters * DoG channels)
        
        # Calculate output dimensions (90x90)
        self.output_height = int(np.sqrt(output_size))
        self.output_width = int(np.sqrt(output_size))
        
        # Calculate receptive field size
        self.receptive_field_size = 2 * receptive_field_radius + 1  # 7x7 for radius 3
        
        # Calculate stride to cover the input image with output neurons
        self.stride_h = self.input_height / self.output_height
        self.stride_w = self.input_width / self.output_width
        
        # Calculate overlap between receptive fields
        self.overlap_h = self.receptive_field_size - self.stride_h
        self.overlap_w = self.receptive_field_size - self.stride_w
        
        # Calculate the size of each receptive field
        self.receptive_field_pixels = self.receptive_field_size * self.receptive_field_size
        
        # Calculate the actual size of each neuron's input
        self.neuron_input_size = self.num_channels * self.receptive_field_pixels
        
        # Initialize weights for each neuron's receptive field
        # Each neuron has weights for all channels in its receptive field
        self.weights = nn.Parameter(torch.randn(output_size, 
                                              self.neuron_input_size, 
                                              device=device))
        self.bias = nn.Parameter(torch.zeros(output_size, device=device))
        
        # Initialize lateral inhibition weights
        self.lateral_weights = nn.Parameter(torch.zeros(output_size, output_size, device=device))
        self.lateral_weights.data.fill_diagonal_(0)
        self.lateral_weights.data.fill_(-0.1)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(output_size).to(device)
        
        # Hebbian learning parameters
        self.eta = learning_rate
        self.decay = 0.01
        
        # Lateral inhibition parameters
        self.inhibition_strength = 0.1
        self.inhibition_decay = 0.01
        self.competition_threshold = 0.1
        
        # Make parameters non-trainable for BP
        self.weights.requires_grad = False
        self.bias.requires_grad = False
        self.lateral_weights.requires_grad = False
        
        # Create spatial mapping for receptive fields
        self.create_receptive_field_mapping()
        
        print(f"L1 Layer initialized with:")
        print(f"- Input shape: {input_shape}")
        print(f"- Output size: {output_size}")
        print(f"- Receptive field size: {self.receptive_field_size}x{self.receptive_field_size}")
        print(f"- Number of channels: {self.num_channels}")
        print(f"- Neuron input size: {self.neuron_input_size}")
        print(f"- Weight matrix shape: {self.weights.shape}")
        print(f"- Stride: {self.stride_h:.3f}x{self.stride_w:.3f} pixels")
        print(f"- Receptive field overlap: {self.overlap_h:.3f}x{self.overlap_w:.3f} pixels")
        print(f"- Each input pixel is covered by approximately {int((self.receptive_field_size/self.stride_h) * (self.receptive_field_size/self.stride_w))} receptive fields")
    
    def create_receptive_field_mapping(self):
        """Create mapping of which input pixels each neuron receives input from."""
        self.receptive_field_mapping = []
        
        # Calculate the center positions for each output neuron
        for i in range(self.output_height):
            for j in range(self.output_width):
                # Calculate the center position in input space
                center_i = int(i * self.stride_h + self.stride_h / 2)
                center_j = int(j * self.stride_w + self.stride_w / 2)
                
                # Calculate the region this neuron receives input from
                start_i = max(0, center_i - self.receptive_field_radius)
                end_i = min(self.input_height, center_i + self.receptive_field_radius + 1)
                start_j = max(0, center_j - self.receptive_field_radius)
                end_j = min(self.input_width, center_j + self.receptive_field_radius + 1)
                
                # Create mask for this receptive field
                mask = torch.zeros((self.input_height, self.input_width), device=self.device)
                mask[start_i:end_i, start_j:end_j] = 1
                
                # Store the mask and its center position
                self.receptive_field_mapping.append({
                    'mask': mask,
                    'center': (center_i, center_j)
                })
    
    def extract_local_receptive_field(self, x, neuron_idx):
        """Extract the local receptive field for a specific neuron."""
        # Get the mask for this neuron
        mask = self.receptive_field_mapping[neuron_idx]['mask']
        
        # Reshape input to 2D spatial dimensions using reshape instead of view
        batch_size = x.size(0)
        x_reshaped = x.reshape(batch_size, self.num_channels, self.input_height, self.input_width)
        
        # Apply mask to get local receptive field
        local_field = x_reshaped * mask.unsqueeze(0).unsqueeze(0)
        
        # Extract only the non-zero region of the receptive field
        center_i, center_j = self.receptive_field_mapping[neuron_idx]['center']
        start_i = max(0, center_i - self.receptive_field_radius)
        end_i = min(self.input_height, center_i + self.receptive_field_radius + 1)
        start_j = max(0, center_j - self.receptive_field_radius)
        end_j = min(self.input_width, center_j + self.receptive_field_radius + 1)
        
        # Extract the actual receptive field region
        local_field = local_field[:, :, start_i:end_i, start_j:end_j]
        
        # Ensure the local field has the correct size
        if local_field.size(2) != self.receptive_field_size or local_field.size(3) != self.receptive_field_size:
            # Pad or crop to the correct size
            padded_field = torch.zeros((batch_size, self.num_channels, 
                                     self.receptive_field_size, self.receptive_field_size), 
                                     device=self.device)
            padded_field[:, :, :local_field.size(2), :local_field.size(3)] = local_field
            local_field = padded_field
        
        # Flatten the local field using reshape instead of view
        return local_field.reshape(batch_size, -1)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Process each neuron's local receptive field
        outputs = []
        for i in range(self.weights.size(0)):
            # Get local receptive field for this neuron
            local_input = self.extract_local_receptive_field(x, i)
            
            # Verify input dimensions
            assert local_input.size(1) == self.neuron_input_size, \
                f"Input size mismatch: got {local_input.size(1)}, expected {self.neuron_input_size}"
            
            # Apply weights to local receptive field
            output = F.linear(local_input, self.weights[i:i+1], self.bias[i:i+1])
            outputs.append(output)
        
        # Stack all neuron outputs
        x = torch.cat(outputs, dim=1)
        
        # Layer normalization
        x = self.layer_norm(x)
        
        # L2 normalization
        x = F.normalize(x, p=2, dim=1)
        
        # Apply competitive lateral inhibition
        inhibition = torch.matmul(x, self.lateral_weights)
        inhibition = torch.where(torch.abs(inhibition) < self.competition_threshold,
                               torch.zeros_like(inhibition),
                               inhibition)
        x = x + self.inhibition_strength * inhibition
        
        return x
    
    def hebbian_update(self, x, y):
        """Update weights using Hebbian learning rule for local receptive fields."""
        for i in range(self.weights.size(0)):
            # Get local receptive field for this neuron
            local_input = self.extract_local_receptive_field(x, i)
            
            # Compute weight updates using Hebbian rule
            weight_update = self.eta * (torch.matmul(y[:, i:i+1].t(), local_input) - 
                                      self.decay * self.weights[i:i+1])
            
            # Update weights
            with torch.no_grad():
                self.weights[i:i+1] += weight_update
        
        # Update lateral inhibition weights
        lateral_update = self.eta * (torch.matmul(y.t(), y) - self.inhibition_decay * self.lateral_weights)
        lateral_update.fill_diagonal_(0)
        lateral_update = torch.where(torch.abs(lateral_update) < self.competition_threshold,
                                   torch.zeros_like(lateral_update),
                                   lateral_update)
        
        with torch.no_grad():
            self.lateral_weights.data += lateral_update

class L2Layer(nn.Module):
    def __init__(self, input_size, output_size, device='cpu', learning_rate=0.000001):
        super(L2Layer, self).__init__()
        self.device = device
        self.learning_rate = learning_rate
        
        # Initialize weights and bias
        self.weights = nn.Parameter(torch.randn(output_size, input_size, device=device))
        self.bias = nn.Parameter(torch.zeros(output_size, device=device))
        
        # Initialize lateral inhibition weights
        self.lateral_weights = nn.Parameter(torch.zeros(output_size, output_size, device=device))
        # Set diagonal to 0 to prevent self-inhibition
        self.lateral_weights.data.fill_diagonal_(0)
        # Initialize with small negative values for inhibition
        self.lateral_weights.data.fill_(-0.1)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(output_size).to(device)
        
        # Hebbian learning parameters
        self.eta = learning_rate  # Learning rate for Hebbian updates
        self.decay = 0.01  # Weight decay factor
        
        # Lateral inhibition parameters
        self.inhibition_strength = 0.1
        self.inhibition_decay = 0.01
        self.competition_threshold = 0.1  # Threshold for competitive inhibition
        
        # Make parameters non-trainable for BP
        self.weights.requires_grad = False
        self.bias.requires_grad = False
        self.lateral_weights.requires_grad = False
    
    def hebbian_update(self, x, y):
        """Update weights using Hebbian learning rule"""
        # Compute weight updates using Hebbian rule: Δw = η * (y * x^T - decay * w)
        weight_update = self.eta * (torch.matmul(y.t(), x) - self.decay * self.weights)
        
        # Update lateral inhibition weights based on competitive learning
        # Stronger activations lead to stronger inhibition of other neurons
        lateral_update = self.eta * (torch.matmul(y.t(), y) - self.inhibition_decay * self.lateral_weights)
        # Ensure diagonal remains 0
        lateral_update.fill_diagonal_(0)
        # Apply competitive threshold
        lateral_update = torch.where(torch.abs(lateral_update) < self.competition_threshold, 
                                   torch.zeros_like(lateral_update), 
                                   lateral_update)
        
        # Apply updates
        with torch.no_grad():
            self.weights.data += weight_update
            self.lateral_weights.data += lateral_update
    
    def forward(self, x):
        # Linear transformation
        x = F.linear(x, self.weights, self.bias)
        
        # Layer normalization
        x = self.layer_norm(x)
        
        # L2 normalization (Euclidean normalization)
        x = F.normalize(x, p=2, dim=1)
        
        # Apply competitive lateral inhibition after normalization
        inhibition = torch.matmul(x, self.lateral_weights)
        # Apply competitive threshold
        inhibition = torch.where(torch.abs(inhibition) < self.competition_threshold,
                               torch.zeros_like(inhibition),
                               inhibition)
        x = x + self.inhibition_strength * inhibition
        
        return x

class L3Layer(nn.Module):
    def __init__(self, input_size, output_size, device='cpu', learning_rate=0.000001):
        super(L3Layer, self).__init__()
        self.device = device
        self.learning_rate = learning_rate
        
        # Initialize weights and bias
        self.weights = nn.Parameter(torch.randn(output_size, input_size, device=device))
        self.bias = nn.Parameter(torch.zeros(output_size, device=device))
        
        # Initialize lateral inhibition weights
        self.lateral_weights = nn.Parameter(torch.zeros(output_size, output_size, device=device))
        # Set diagonal to 0 to prevent self-inhibition
        self.lateral_weights.data.fill_diagonal_(0)
        # Initialize with small negative values for inhibition
        self.lateral_weights.data.fill_(-0.1)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(output_size).to(device)
        
        # Hebbian learning parameters
        self.eta = learning_rate  # Learning rate for Hebbian updates
        self.decay = 0.01  # Weight decay factor
        
        # Lateral inhibition parameters
        self.inhibition_strength = 0.1
        self.inhibition_decay = 0.01
        self.competition_threshold = 0.1  # Threshold for competitive inhibition
        
        # Make parameters non-trainable for BP
        self.weights.requires_grad = False
        self.bias.requires_grad = False
        self.lateral_weights.requires_grad = False
    
    def hebbian_update(self, x, y):
        """Update weights using Hebbian learning rule"""
        # Compute weight updates using Hebbian rule: Δw = η * (y * x^T - decay * w)
        weight_update = self.eta * (torch.matmul(y.t(), x) - self.decay * self.weights)
        
        # Update lateral inhibition weights based on competitive learning
        # Stronger activations lead to stronger inhibition of other neurons
        lateral_update = self.eta * (torch.matmul(y.t(), y) - self.inhibition_decay * self.lateral_weights)
        # Ensure diagonal remains 0
        lateral_update.fill_diagonal_(0)
        # Apply competitive threshold
        lateral_update = torch.where(torch.abs(lateral_update) < self.competition_threshold, 
                                   torch.zeros_like(lateral_update), 
                                   lateral_update)
        
        # Apply updates
        with torch.no_grad():
            self.weights.data += weight_update
            self.lateral_weights.data += lateral_update
    
    def forward(self, x):
        # Linear transformation
        x = F.linear(x, self.weights, self.bias)
        
        # Layer normalization
        x = self.layer_norm(x)
        
        # L2 normalization (Euclidean normalization)
        x = F.normalize(x, p=2, dim=1)
        
        # Apply competitive lateral inhibition after normalization
        inhibition = torch.matmul(x, self.lateral_weights)
        # Apply competitive threshold
        inhibition = torch.where(torch.abs(inhibition) < self.competition_threshold,
                               torch.zeros_like(inhibition),
                               inhibition)
        x = x + self.inhibition_strength * inhibition
        
        return x

class L4Layer(nn.Module):
    def __init__(self, input_size, output_size, device='cpu', learning_rate=0.000001):
        super(L4Layer, self).__init__()
        self.device = device
        self.learning_rate = learning_rate
        
        # Initialize weights and bias
        self.weights = nn.Parameter(torch.randn(output_size, input_size, device=device))
        self.bias = nn.Parameter(torch.zeros(output_size, device=device))
        
        # Initialize lateral inhibition weights
        self.lateral_weights = nn.Parameter(torch.zeros(output_size, output_size, device=device))
        # Set diagonal to 0 to prevent self-inhibition
        self.lateral_weights.data.fill_diagonal_(0)
        # Initialize with small negative values for inhibition
        self.lateral_weights.data.fill_(-0.1)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(output_size).to(device)
        
        # Hebbian learning parameters
        self.eta = learning_rate  # Learning rate for Hebbian updates
        self.decay = 0.01  # Weight decay factor
        
        # Lateral inhibition parameters
        self.inhibition_strength = 0.1
        self.inhibition_decay = 0.01
        self.competition_threshold = 0.1  # Threshold for competitive inhibition
        
        # Make parameters non-trainable for BP
        self.weights.requires_grad = False
        self.bias.requires_grad = False
        self.lateral_weights.requires_grad = False
    
    def hebbian_update(self, x, y):
        """Update weights using Hebbian learning rule"""
        # Compute weight updates using Hebbian rule: Δw = η * (y * x^T - decay * w)
        weight_update = self.eta * (torch.matmul(y.t(), x) - self.decay * self.weights)
        
        # Update lateral inhibition weights based on competitive learning
        # Stronger activations lead to stronger inhibition of other neurons
        lateral_update = self.eta * (torch.matmul(y.t(), y) - self.inhibition_decay * self.lateral_weights)
        # Ensure diagonal remains 0
        lateral_update.fill_diagonal_(0)
        # Apply competitive threshold
        lateral_update = torch.where(torch.abs(lateral_update) < self.competition_threshold, 
                                   torch.zeros_like(lateral_update), 
                                   lateral_update)
        
        # Apply updates
        with torch.no_grad():
            self.weights.data += weight_update
            self.lateral_weights.data += lateral_update
    
    def forward(self, x):
        # Linear transformation
        x = F.linear(x, self.weights, self.bias)
        
        # Layer normalization
        x = self.layer_norm(x)
        
        # L2 normalization (Euclidean normalization)
        x = F.normalize(x, p=2, dim=1)
        
        # Apply competitive lateral inhibition after normalization
        inhibition = torch.matmul(x, self.lateral_weights)
        # Apply competitive threshold
        inhibition = torch.where(torch.abs(inhibition) < self.competition_threshold,
                               torch.zeros_like(inhibition),
                               inhibition)
        x = x + self.inhibition_strength * inhibition
        
        return x

class VisNetWithDoGAndGabor(nn.Module):
    def __init__(self, input_shape, l1_size, l2_size, l3_size, l4_size, num_classes=10, device='cpu'):
        super(VisNetWithDoGAndGabor, self).__init__()
        self.device = device
        
        # Initialize DoG filter
        self.dog_filter = create_dog_filter(sigma1=1.0, sigma2=1.2, size=3, device=device)
        
        # Initialize Gabor filters
        self.gabor_filters_real, self.gabor_filters_imag = create_gabor_filters(device=device)
        
        # Calculate input size for L1 layer
        num_filters = self.gabor_filters_real.size(0)  # Number of Gabor filters
        num_dog_channels = 3  # Luminance, R-G, B-G
        self.num_channels = num_filters * num_dog_channels  # Total number of channels after filtering
        
        print(f"Input shape: {input_shape}")
        print(f"Number of Gabor filters: {num_filters}")
        print(f"Number of DoG channels: {num_dog_channels}")
        print(f"Total number of channels: {self.num_channels}")
        
        # Create L1, L2, L3, and L4 layers with Hebbian learning
        self.l1 = L1Layer((self.num_channels, input_shape[1], input_shape[2]), l1_size, device)
        self.l2 = L2Layer(l1_size, l2_size, device)
        self.l3 = L3Layer(l2_size, l3_size, device)
        self.l4 = L4Layer(l3_size, l4_size, device)
        
        # Add single linear classification layer (BP only)
        self.classifier = nn.Linear(l4_size, num_classes).to(device)
        
        # Initialize classifier weights
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Apply DoG filter to get multiple channels (luminance, R-G, B-G)
        x = apply_dog_filter(x, self.dog_filter)
        
        # Apply Gabor filters to each channel
        filtered_outputs = []
        for channel in range(x.size(1)):
            channel_input = x[:, channel:channel+1]
            # Apply both real and imaginary filters
            filtered_real = F.conv2d(channel_input, self.gabor_filters_real, padding='same')
            filtered_imag = F.conv2d(channel_input, self.gabor_filters_imag, padding='same')
            # Combine real and imaginary parts
            filtered = torch.sqrt(filtered_real**2 + filtered_imag**2)
            filtered_outputs.append(filtered)
        
        # Concatenate filtered outputs from all channels
        x = torch.cat(filtered_outputs, dim=1)
        
        # Pass through L1 layer and get intermediate representation
        l1_output = self.l1(x)
        
        # Apply Hebbian learning to L1 layer
        self.l1.hebbian_update(x, l1_output)
        
        # Detach L1 output to prevent BP from affecting L1
        l1_output = l1_output.detach()
        
        # Pass through L2 layer
        l2_output = self.l2(l1_output)
        
        # Apply Hebbian learning to L2 layer
        self.l2.hebbian_update(l1_output, l2_output)
        
        # Detach L2 output to prevent BP from affecting L2
        l2_output = l2_output.detach()
        
        # Pass through L3 layer
        l3_output = self.l3(l2_output)
        
        # Apply Hebbian learning to L3 layer
        self.l3.hebbian_update(l2_output, l3_output)
        
        # Detach L3 output to prevent BP from affecting L3
        l3_output = l3_output.detach()
        
        # Pass through L4 layer
        l4_output = self.l4(l3_output)
        
        # Apply Hebbian learning to L4 layer
        self.l4.hebbian_update(l3_output, l4_output)
        
        # Detach L4 output to prevent BP from affecting L4
        l4_output = l4_output.detach()
        
        # Classification (BP only)
        x = self.classifier(l4_output)
        
        return x

def visualize_dog_and_gabor_filters(model, images, device, num_images=4):
    """Visualize DoG and Gabor filter responses to input images."""
    model.eval()
    with torch.no_grad():
        # Get first batch of images
        images = images[:num_images].to(device)
        
        # Apply DoG filter
        dog_filtered = apply_dog_filter(images, model.dog_filter)
        
        # Get filtered outputs for each channel
        filtered_outputs = []
        for channel in range(dog_filtered.size(1)):
            channel_input = dog_filtered[:, channel:channel+1]
            # Apply both real and imaginary filters
            filtered_real = F.conv2d(channel_input, model.gabor_filters_real, padding='same')
            filtered_imag = F.conv2d(channel_input, model.gabor_filters_imag, padding='same')
            # Combine real and imaginary parts
            filtered = torch.sqrt(filtered_real**2 + filtered_imag**2)
            filtered_outputs.append(filtered)
        
        # Concatenate filtered outputs
        gabor_filtered = torch.cat(filtered_outputs, dim=1)
        
        # Visualize original, DoG filtered, and Gabor filtered images
        plt.figure(figsize=(15, 5*num_images))
        for i in range(num_images):
            # Original image
            plt.subplot(num_images, 3, 3*i+1)
            plt.imshow(images[i].permute(1, 2, 0).cpu().numpy())
            plt.title('Original Image')
            plt.axis('off')
            
            # DoG filtered image
            plt.subplot(num_images, 3, 3*i+2)
            plt.imshow(dog_filtered[i, 0].cpu().numpy(), cmap='gray')
            plt.title('DoG Filtered')
            plt.axis('off')
            
            # Gabor filtered image (show first channel)
            plt.subplot(num_images, 3, 3*i+3)
            plt.imshow(gabor_filtered[i, 0].cpu().numpy(), cmap='gray')
            plt.title('Gabor Filtered')
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('dog_and_gabor_filtered_images.png')
        plt.show()

def train(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc='Training')):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        output = model(data)
        loss = F.cross_entropy(output, target)
        
        # Backward pass (only affects classifier)
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        total_loss += loss.item()
    
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(train_loader)
    return avg_loss, accuracy

def test(model, test_loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc='Testing'):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target).item()
            
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    accuracy = 100. * correct / total
    avg_loss = test_loss / len(test_loader)
    return avg_loss, accuracy

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load CIFAR-10 dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # Initialize model
    input_shape = (3, 32, 32)  # CIFAR-10 images are 32x32 with 3 channels
    layer_size = 8100
    num_classes = 10  # CIFAR-10 has 10 classes
    
    model = VisNetWithDoGAndGabor(input_shape, layer_size, layer_size, layer_size, layer_size, num_classes, device)
    model = model.to(device)
    
    # Get a batch of images and visualize filtered results
    images, _ = next(iter(train_loader))
    visualize_dog_and_gabor_filters(model, images, device)
    
    # Initialize optimizer (only for classifier)
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 100
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss, train_acc = train(model, train_loader, optimizer, device)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # Test
        test_loss, test_acc = test(model, test_loader, device)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
    
    # Plot results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()

if __name__ == "__main__":
    main() 