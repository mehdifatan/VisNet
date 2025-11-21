import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from typing import List, Tuple

# ============================================================================
# LOCAL INHIBITION & SPARSENESS (VisNet-LI style)
# ============================================================================

def apply_local_inhibition_and_sparseness(x, sparsity, layer_radius, layer_size, device):
    """
    Apply local inhibition and sparseness to layer activations.
    
    Args:
        x: Activations [batch_size, num_neurons]
        sparsity: Target sparsity (e.g., 0.99 keeps top 1%)
        layer_radius: Radius for local inhibition kernel
        layer_size: Size of layer grid (e.g., 90 for 90x90 = 8100 neurons)
        device: torch device
    
    Returns:
        Inhibited and sparse activations
    """
    B = x.size(0)
    # Reshape to 2D grid
    xg = x.view(B, 1, layer_size, layer_size)
    
    # Create Gaussian inhibition kernel
    yy, xx = torch.meshgrid(
        torch.arange(-layer_radius, layer_radius + 1, device=device),
        torch.arange(-layer_radius, layer_radius + 1, device=device), 
        indexing='ij')
    dist = torch.sqrt(xx**2 + yy**2)
    k = torch.exp(-0.5 * (dist / layer_radius) ** 2)
    k /= k.sum()
    
    # Apply local inhibition: subtract weighted average of neighbors
    inhibited = xg - 0.1 * (F.conv2d(xg, k.unsqueeze(0).unsqueeze(0), padding=layer_radius) - xg)
    inhibited = inhibited.view(B, -1)
    
    # Apply sparseness: keep top (1-sparsity) * 100% of neurons
    keep = int((1 - sparsity) * inhibited.size(1))
    if keep > 0:
        vals, _ = torch.topk(inhibited, keep, dim=1)
        thr = vals[:, -1].unsqueeze(1)
        inhibited = torch.where(inhibited >= thr, inhibited, torch.zeros_like(inhibited))
    
    return inhibited

# ============================================================================
# FILTERS
# ============================================================================

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
        self.num_channels = input_shape[0]  # Number of input channels
        
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
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(output_size).to(device)
        
        # Hebbian learning parameters
        self.eta = learning_rate
        self.decay = 0.01
        
        # Local inhibition parameters (VisNet-LI style)
        self.sparsity = 0.99  # Keep top 1% active
        self.inhibition_radius = 3  # Radius for local inhibition kernel
        
        # Make parameters non-trainable for BP
        self.weights.requires_grad = False
        self.bias.requires_grad = False
        
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
        
        # Apply ReLU activation
        x = F.relu(x)
        
        # Apply local inhibition and sparseness (VisNet-LI style)
        x = apply_local_inhibition_and_sparseness(
            x, self.sparsity, self.inhibition_radius, self.output_height, self.device
        )
        
        # Layer normalization
        x = self.layer_norm(x)
        
        return x
    
    def hebbian_update(self, x, y):
        """Update weights using Hebbian learning rule for local receptive fields."""
        for i in range(self.weights.size(0)):
            # Get local receptive field for this neuron
            local_input = self.extract_local_receptive_field(x, i)
            
            yi = y[:, i:i+1]
            # Outer product between activation and receptive field
            xy = torch.matmul(yi.t(), local_input)
            yy = torch.matmul(yi.t(), yi)
            weight_update = self.eta * (xy - yy * self.weights[i:i+1])
            
            # Update weights
            with torch.no_grad():
                self.weights[i:i+1] += weight_update

class L2Layer(nn.Module):
    def __init__(self, input_size, output_size, device='cpu', learning_rate=0.000001, receptive_field_radius=3):
        super(L2Layer, self).__init__()
        self.device = device
        self.learning_rate = learning_rate
        self.output_size = output_size
        self.input_size = input_size
        self.receptive_field_radius = receptive_field_radius
        
        # Calculate layer dimensions for 2D grid (both input and output)
        self.layer_size = int(np.sqrt(output_size))
        self.input_layer_size = int(np.sqrt(input_size))
        
        # Calculate receptive field size
        self.receptive_field_size = 2 * receptive_field_radius + 1  # 7x7 for radius 3
        self.receptive_field_neurons = self.receptive_field_size * self.receptive_field_size
        
        # Initialize weights and bias with SPATIAL receptive fields
        self.weights = nn.Parameter(torch.randn(output_size, self.receptive_field_neurons, device=device))
        self.bias = nn.Parameter(torch.zeros(output_size, device=device))
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(output_size).to(device)
        
        # Hebbian learning parameters
        self.eta = learning_rate  # Learning rate for Hebbian updates
        self.decay = 0.01  # Weight decay factor
        
        # Local inhibition parameters (VisNet-LI style)
        self.sparsity = 0.99  # Keep top 1% active
        self.inhibition_radius = 3  # Radius for local inhibition kernel
        
        # Make parameters non-trainable for BP
        self.weights.requires_grad = False
        self.bias.requires_grad = False
        
        # Create receptive field mapping
        self.create_receptive_field_mapping()
        
        print(f"L2 Layer initialized:")
        print(f"- Input size: {input_size} ({self.input_layer_size}x{self.input_layer_size})")
        print(f"- Output size: {output_size} ({self.layer_size}x{self.layer_size})")
        print(f"- Receptive field: {self.receptive_field_size}x{self.receptive_field_size}")
        print(f"- Weights per neuron: {self.receptive_field_neurons}")
    
    def create_receptive_field_mapping(self):
        """Create mapping of which input neurons each output neuron connects to."""
        self.receptive_field_mapping = []
        
        for i in range(self.layer_size):
            for j in range(self.layer_size):
                # Map output position to input position (assuming same spatial layout)
                center_i = int(i * self.input_layer_size / self.layer_size)
                center_j = int(j * self.input_layer_size / self.layer_size)
                
                # Calculate receptive field bounds
                start_i = max(0, center_i - self.receptive_field_radius)
                end_i = min(self.input_layer_size, center_i + self.receptive_field_radius + 1)
                start_j = max(0, center_j - self.receptive_field_radius)
                end_j = min(self.input_layer_size, center_j + self.receptive_field_radius + 1)
                
                self.receptive_field_mapping.append({
                    'center': (center_i, center_j),
                    'bounds': (start_i, end_i, start_j, end_j)
                })
    
    def extract_local_receptive_field(self, x, neuron_idx):
        """Extract local receptive field for a specific neuron."""
        batch_size = x.size(0)
        
        # Reshape input to 2D grid [batch, input_layer_size, input_layer_size]
        x_2d = x.view(batch_size, self.input_layer_size, self.input_layer_size)
        
        # Get receptive field bounds
        start_i, end_i, start_j, end_j = self.receptive_field_mapping[neuron_idx]['bounds']
        
        # Extract receptive field
        local_field = x_2d[:, start_i:end_i, start_j:end_j]
        
        # Pad if necessary to ensure consistent size
        if local_field.size(1) != self.receptive_field_size or local_field.size(2) != self.receptive_field_size:
            padded_field = torch.zeros((batch_size, self.receptive_field_size, self.receptive_field_size), 
                                       device=self.device)
            padded_field[:, :local_field.size(1), :local_field.size(2)] = local_field
            local_field = padded_field
        
        # Flatten to [batch, receptive_field_neurons]
        return local_field.reshape(batch_size, -1)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Process each neuron's local receptive field
        outputs = []
        for i in range(self.weights.size(0)):
            # Get local receptive field for this neuron
            local_input = self.extract_local_receptive_field(x, i)
            
            # Apply weights to local receptive field
            output = F.linear(local_input, self.weights[i:i+1], self.bias[i:i+1])
            outputs.append(output)
        
        # Stack all neuron outputs
        x = torch.cat(outputs, dim=1)
        
        # Apply ReLU activation
        x = F.relu(x)
        
        # Apply local inhibition and sparseness (VisNet-LI style)
        x = apply_local_inhibition_and_sparseness(
            x, self.sparsity, self.inhibition_radius, self.layer_size, self.device
        )
        
        # Layer normalization
        x = self.layer_norm(x)
        
        return x
    
    def hebbian_update(self, x, y):
        """Update weights using Hebbian learning rule for local receptive fields."""
        for i in range(self.weights.size(0)):
            # Get local receptive field for this neuron
            local_input = self.extract_local_receptive_field(x, i)
            
            # Compute weight updates using Hebbian rule
            yi = y[:, i:i+1]
            xy = torch.matmul(yi.t(), local_input)
            yy = torch.matmul(yi.t(), yi)
            weight_update = self.eta * (xy - yy * self.weights[i:i+1])
            
            # Update weights
            with torch.no_grad():
                self.weights[i:i+1] += weight_update

class L3Layer(nn.Module):
    def __init__(self, input_size, output_size, device='cpu', learning_rate=0.000001, receptive_field_radius=3):
        super(L3Layer, self).__init__()
        self.device = device
        self.learning_rate = learning_rate
        self.output_size = output_size
        self.input_size = input_size
        self.receptive_field_radius = receptive_field_radius
        
        # Calculate layer dimensions for 2D grid (both input and output)
        self.layer_size = int(np.sqrt(output_size))
        self.input_layer_size = int(np.sqrt(input_size))
        
        # Calculate receptive field size
        self.receptive_field_size = 2 * receptive_field_radius + 1  # 7x7 for radius 3
        self.receptive_field_neurons = self.receptive_field_size * self.receptive_field_size
        
        # Initialize weights and bias with SPATIAL receptive fields
        self.weights = nn.Parameter(torch.randn(output_size, self.receptive_field_neurons, device=device))
        self.bias = nn.Parameter(torch.zeros(output_size, device=device))
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(output_size).to(device)
        
        # Hebbian learning parameters
        self.eta = learning_rate  # Learning rate for Hebbian updates
        self.decay = 0.01  # Weight decay factor
        
        # Local inhibition parameters (VisNet-LI style)
        self.sparsity = 0.99  # Keep top 1% active
        self.inhibition_radius = 3  # Radius for local inhibition kernel
        
        # Make parameters non-trainable for BP
        self.weights.requires_grad = False
        self.bias.requires_grad = False
        
        # Create receptive field mapping
        self.create_receptive_field_mapping()
        
        print(f"L3 Layer initialized:")
        print(f"- Input size: {input_size} ({self.input_layer_size}x{self.input_layer_size})")
        print(f"- Output size: {output_size} ({self.layer_size}x{self.layer_size})")
        print(f"- Receptive field: {self.receptive_field_size}x{self.receptive_field_size}")
        print(f"- Weights per neuron: {self.receptive_field_neurons}")
    
    def create_receptive_field_mapping(self):
        """Create mapping of which input neurons each output neuron connects to."""
        self.receptive_field_mapping = []
        
        for i in range(self.layer_size):
            for j in range(self.layer_size):
                # Map output position to input position (assuming same spatial layout)
                center_i = int(i * self.input_layer_size / self.layer_size)
                center_j = int(j * self.input_layer_size / self.layer_size)
                
                # Calculate receptive field bounds
                start_i = max(0, center_i - self.receptive_field_radius)
                end_i = min(self.input_layer_size, center_i + self.receptive_field_radius + 1)
                start_j = max(0, center_j - self.receptive_field_radius)
                end_j = min(self.input_layer_size, center_j + self.receptive_field_radius + 1)
                
                self.receptive_field_mapping.append({
                    'center': (center_i, center_j),
                    'bounds': (start_i, end_i, start_j, end_j)
                })
    
    def extract_local_receptive_field(self, x, neuron_idx):
        """Extract local receptive field for a specific neuron."""
        batch_size = x.size(0)
        
        # Reshape input to 2D grid [batch, input_layer_size, input_layer_size]
        x_2d = x.view(batch_size, self.input_layer_size, self.input_layer_size)
        
        # Get receptive field bounds
        start_i, end_i, start_j, end_j = self.receptive_field_mapping[neuron_idx]['bounds']
        
        # Extract receptive field
        local_field = x_2d[:, start_i:end_i, start_j:end_j]
        
        # Pad if necessary to ensure consistent size
        if local_field.size(1) != self.receptive_field_size or local_field.size(2) != self.receptive_field_size:
            padded_field = torch.zeros((batch_size, self.receptive_field_size, self.receptive_field_size), 
                                       device=self.device)
            padded_field[:, :local_field.size(1), :local_field.size(2)] = local_field
            local_field = padded_field
        
        # Flatten to [batch, receptive_field_neurons]
        return local_field.reshape(batch_size, -1)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Process each neuron's local receptive field
        outputs = []
        for i in range(self.weights.size(0)):
            # Get local receptive field for this neuron
            local_input = self.extract_local_receptive_field(x, i)
            
            # Apply weights to local receptive field
            output = F.linear(local_input, self.weights[i:i+1], self.bias[i:i+1])
            outputs.append(output)
        
        # Stack all neuron outputs
        x = torch.cat(outputs, dim=1)
        
        # Apply ReLU activation
        x = F.relu(x)
        
        # Apply local inhibition and sparseness (VisNet-LI style)
        x = apply_local_inhibition_and_sparseness(
            x, self.sparsity, self.inhibition_radius, self.layer_size, self.device
        )
        
        # Layer normalization
        x = self.layer_norm(x)
        
        return x
    
    def hebbian_update(self, x, y):
        """Update weights using Hebbian learning rule for local receptive fields."""
        for i in range(self.weights.size(0)):
            # Get local receptive field for this neuron
            local_input = self.extract_local_receptive_field(x, i)
            
            # Compute weight updates using Hebbian rule
            yi = y[:, i:i+1]
            xy = torch.matmul(yi.t(), local_input)
            yy = torch.matmul(yi.t(), yi)
            weight_update = self.eta * (xy - yy * self.weights[i:i+1])
            
            # Update weights
            with torch.no_grad():
                self.weights[i:i+1] += weight_update

class L4Layer(nn.Module):
    def __init__(self, input_size, output_size, device='cpu', learning_rate=0.000001, receptive_field_radius=3):
        super(L4Layer, self).__init__()
        self.device = device
        self.learning_rate = learning_rate
        self.output_size = output_size
        self.input_size = input_size
        self.receptive_field_radius = receptive_field_radius
        
        # Calculate layer dimensions for 2D grid (both input and output)
        self.layer_size = int(np.sqrt(output_size))
        self.input_layer_size = int(np.sqrt(input_size))
        
        # Calculate receptive field size
        self.receptive_field_size = 2 * receptive_field_radius + 1  # 7x7 for radius 3
        self.receptive_field_neurons = self.receptive_field_size * self.receptive_field_size
        
        # Initialize weights and bias with SPATIAL receptive fields
        self.weights = nn.Parameter(torch.randn(output_size, self.receptive_field_neurons, device=device))
        self.bias = nn.Parameter(torch.zeros(output_size, device=device))
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(output_size).to(device)
        
        # Hebbian learning parameters
        self.eta = learning_rate  # Learning rate for Hebbian updates
        self.decay = 0.01  # Weight decay factor
        
        # Local inhibition parameters (VisNet-LI style)
        self.sparsity = 0.99  # Keep top 1% active
        self.inhibition_radius = 3  # Radius for local inhibition kernel
        
        # Make parameters non-trainable for BP
        self.weights.requires_grad = False
        self.bias.requires_grad = False
        
        # Create receptive field mapping
        self.create_receptive_field_mapping()
        
        print(f"L4 Layer initialized:")
        print(f"- Input size: {input_size} ({self.input_layer_size}x{self.input_layer_size})")
        print(f"- Output size: {output_size} ({self.layer_size}x{self.layer_size})")
        print(f"- Receptive field: {self.receptive_field_size}x{self.receptive_field_size}")
        print(f"- Weights per neuron: {self.receptive_field_neurons}")
    
    def create_receptive_field_mapping(self):
        """Create mapping of which input neurons each output neuron connects to."""
        self.receptive_field_mapping = []
        
        for i in range(self.layer_size):
            for j in range(self.layer_size):
                # Map output position to input position (assuming same spatial layout)
                center_i = int(i * self.input_layer_size / self.layer_size)
                center_j = int(j * self.input_layer_size / self.layer_size)
                
                # Calculate receptive field bounds
                start_i = max(0, center_i - self.receptive_field_radius)
                end_i = min(self.input_layer_size, center_i + self.receptive_field_radius + 1)
                start_j = max(0, center_j - self.receptive_field_radius)
                end_j = min(self.input_layer_size, center_j + self.receptive_field_radius + 1)
                
                self.receptive_field_mapping.append({
                    'center': (center_i, center_j),
                    'bounds': (start_i, end_i, start_j, end_j)
                })
    
    def extract_local_receptive_field(self, x, neuron_idx):
        """Extract local receptive field for a specific neuron."""
        batch_size = x.size(0)
        
        # Reshape input to 2D grid [batch, input_layer_size, input_layer_size]
        x_2d = x.view(batch_size, self.input_layer_size, self.input_layer_size)
        
        # Get receptive field bounds
        start_i, end_i, start_j, end_j = self.receptive_field_mapping[neuron_idx]['bounds']
        
        # Extract receptive field
        local_field = x_2d[:, start_i:end_i, start_j:end_j]
        
        # Pad if necessary to ensure consistent size
        if local_field.size(1) != self.receptive_field_size or local_field.size(2) != self.receptive_field_size:
            padded_field = torch.zeros((batch_size, self.receptive_field_size, self.receptive_field_size), 
                                       device=self.device)
            padded_field[:, :local_field.size(1), :local_field.size(2)] = local_field
            local_field = padded_field
        
        # Flatten to [batch, receptive_field_neurons]
        return local_field.reshape(batch_size, -1)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Process each neuron's local receptive field
        outputs = []
        for i in range(self.weights.size(0)):
            # Get local receptive field for this neuron
            local_input = self.extract_local_receptive_field(x, i)
            
            # Apply weights to local receptive field
            output = F.linear(local_input, self.weights[i:i+1], self.bias[i:i+1])
            outputs.append(output)
        
        # Stack all neuron outputs
        x = torch.cat(outputs, dim=1)
        
        # Apply ReLU activation
        x = F.relu(x)
        
        # Apply local inhibition and sparseness (VisNet-LI style)
        x = apply_local_inhibition_and_sparseness(
            x, self.sparsity, self.inhibition_radius, self.layer_size, self.device
        )
        
        # Layer normalization
        x = self.layer_norm(x)
        
        return x
    
    def hebbian_update(self, x, y):
        """Update weights using Hebbian learning rule for local receptive fields."""
        for i in range(self.weights.size(0)):
            # Get local receptive field for this neuron
            local_input = self.extract_local_receptive_field(x, i)
            
            # Compute weight updates using Hebbian rule
            yi = y[:, i:i+1]
            xy = torch.matmul(yi.t(), local_input)
            yy = torch.matmul(yi.t(), yi)
            weight_update = self.eta * (xy - yy * self.weights[i:i+1])
            
            # Update weights
            with torch.no_grad():
                self.weights[i:i+1] += weight_update

class VisNetWithGabor(nn.Module):
    def __init__(self, input_shape, l1_size, l2_size, l3_size, l4_size, num_classes=10, device='cpu',
                 hebbian_lr=0.000001, sparsity=0.99, l1_rf_radius=3, l2_rf_radius=3, l3_rf_radius=3, l4_rf_radius=3):
        super(VisNetWithGabor, self).__init__()
        self.device = device

        # Initialize Gabor filters
        self.gabor_filters_real, self.gabor_filters_imag = create_gabor_filters(device=device)

        # Calculate input size for L1 layer
        num_filters = self.gabor_filters_real.size(0)  # Number of Gabor filters
        num_input_channels = input_shape[0]
        self.num_channels = num_filters * num_input_channels  # Total number of channels after filtering
        
        print(f"\n{'='*60}")
        print(f"VisNet Model Configuration")
        print(f"{'='*60}")
        print(f"Input shape: {input_shape}")
        print(f"Number of Gabor filters: {num_filters}")
        print(f"Input channels: {num_input_channels}")
        print(f"Total number of channels: {self.num_channels}")
        print(f"Hebbian learning rate: {hebbian_lr}")
        print(f"Sparsity: {sparsity} (keep top {(1-sparsity)*100:.1f}% active)")
        print(f"Receptive field radii: L1={l1_rf_radius}, L2={l2_rf_radius}, L3={l3_rf_radius}, L4={l4_rf_radius}")
        print(f"{'='*60}\n")
        
        # Create L1, L2, L3, and L4 layers with Hebbian learning and customizable receptive fields
        self.l1 = L1Layer((self.num_channels, input_shape[1], input_shape[2]), l1_size, device, 
                         learning_rate=hebbian_lr, receptive_field_radius=l1_rf_radius)
        self.l2 = L2Layer(l1_size, l2_size, device, learning_rate=hebbian_lr, receptive_field_radius=l2_rf_radius)
        self.l3 = L3Layer(l2_size, l3_size, device, learning_rate=hebbian_lr, receptive_field_radius=l3_rf_radius)
        self.l4 = L4Layer(l3_size, l4_size, device, learning_rate=hebbian_lr, receptive_field_radius=l4_rf_radius)
        
        # Update sparsity for all layers
        self.l1.sparsity = sparsity
        self.l2.sparsity = sparsity
        self.l3.sparsity = sparsity
        self.l4.sparsity = sparsity
        
        # Add single linear classification layer (BP only)
        self.classifier = nn.Linear(l4_size, num_classes).to(device)
        
        # Initialize classifier weights
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Apply DoG filter to get multiple channels (luminance, R-G, B-G)
        # Apply Gabor filters to each color channel
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
        
        # Apply Hebbian learning to L1 layer (ONLY during training)
        if self.training:
            self.l1.hebbian_update(x, l1_output)
        
        # Detach L1 output to prevent BP from affecting L1
        l1_output = l1_output.detach()
        
        # Pass through L2 layer
        l2_output = self.l2(l1_output)
        
        # Apply Hebbian learning to L2 layer (ONLY during training)
        if self.training:
            self.l2.hebbian_update(l1_output, l2_output)
        
        # Detach L2 output to prevent BP from affecting L2
        l2_output = l2_output.detach()
        
        # Pass through L3 layer
        l3_output = self.l3(l2_output)
        
        # Apply Hebbian learning to L3 layer (ONLY during training)
        if self.training:
            self.l3.hebbian_update(l2_output, l3_output)
        
        # Detach L3 output to prevent BP from affecting L3
        l3_output = l3_output.detach()
        
        # Pass through L4 layer
        l4_output = self.l4(l3_output)
        
        # Apply Hebbian learning to L4 layer (ONLY during training)
        if self.training:
            self.l4.hebbian_update(l3_output, l4_output)
        
        # Detach L4 output to prevent BP from affecting L4
        l4_output = l4_output.detach()
        
        # Classification (BP only)
        x = self.classifier(l4_output)
        
        return x

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
    # ========== CONFIGURATION ==========
    BATCH_SIZE = 32              # Batch size (increase for GPU: 64, 128, 256)
    NUM_EPOCHS = 100             # Number of training epochs
    HEBBIAN_LR = 0.001           # Hebbian learning rate (unsupervised) - CRITICAL: don't set too low!
    CLASSIFIER_LR = 0.001        # Classifier learning rate (supervised)
    LAYER_SIZE = 100            # Size of each layer (6400 = 80x80)
    NUM_WORKERS = 4              # Number of data loading workers (set to 0 if issues)
    SPARSITY = 0.95              # Sparsity level (0.95 = keep top 5% active, less sparse than 0.99)
    
    # Set device with detailed GPU info
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n{'='*60}")
    print(f"Training Configuration")
    print(f"{'='*60}")
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"CUDA Version: {torch.version.cuda}")
        # Set memory optimization
        torch.backends.cudnn.benchmark = True
        print(f"cuDNN Benchmark: Enabled")
    else:
        print("WARNING: No GPU detected! Training will be slow on CPU.")
        NUM_WORKERS = 0  # No workers on CPU
    
    print(f"\nHyperparameters:")
    print(f"- Batch size: {BATCH_SIZE}")
    print(f"- Epochs: {NUM_EPOCHS}")
    print(f"- Hebbian LR: {HEBBIAN_LR}")
    print(f"- Classifier LR: {CLASSIFIER_LR}")
    print(f"- Layer size: {LAYER_SIZE} ({int(np.sqrt(LAYER_SIZE))}x{int(np.sqrt(LAYER_SIZE))})")
    print(f"- Sparsity: {SPARSITY} (keep top {(1-SPARSITY)*100:.1f}% active)")
    print(f"- Data workers: {NUM_WORKERS}")
    print(f"{'='*60}\n")
    
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    # Optimize DataLoader for GPU with pin_memory and num_workers
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True if NUM_WORKERS > 0 else False
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True if NUM_WORKERS > 0 else False
    )
    
    # Initialize model
    input_shape = (1, 32, 32)  # MNIST images converted to 32x32 grayscale
    num_classes = 10
    
    model = VisNetWithGabor(
        input_shape, LAYER_SIZE, LAYER_SIZE, LAYER_SIZE, LAYER_SIZE, 
        num_classes, device, hebbian_lr=HEBBIAN_LR, sparsity=SPARSITY
    )
    model = model.to(device)
    
    # Initialize optimizer (only for classifier)
    optimizer = optim.Adam(model.classifier.parameters(), lr=CLASSIFIER_LR)
    
    # Training loop
    for epoch in range(NUM_EPOCHS):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        print(f"{'='*60}")
        
        # Train
        train_loss, train_acc = train(model, train_loader, optimizer, device)
        
        # Test
        test_loss, test_acc = test(model, test_loader, device)
        
        # Print results
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
        
        # Print GPU memory usage if available
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(device) / 1e9
            reserved = torch.cuda.memory_reserved(device) / 1e9
            print(f"GPU Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
            
            # Clear cache periodically to prevent fragmentation
            if (epoch + 1) % 10 == 0:
                torch.cuda.empty_cache()
                print("GPU cache cleared")

if __name__ == "__main__":
    main() 