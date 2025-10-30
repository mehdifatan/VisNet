"""
VisNet: Complete Implementation in Single Script
==================================================

A well-structured, production-ready implementation of VisNet architectures
for biologically-inspired hierarchical vision networks.

Supports:
- VisNet-MD-Linear: Manhattan Distance Learning with Linear activation
- VisNet-RBF-MD: Manhattan Distance Learning with RBF activation
- VisNet-LI: Oja's Hebbian Learning with Local Inhibition
- VisNet-LI-DoG: VisNet-LI with DoG preprocessing
- Simplified VisNet: Oja's Hebbian Learning with Global WTA

Author: Based on Rolls 2015 paper
Version: 1.0.0
"""


# ============================================================================
# SECTION 1: IMPORTS
# ============================================================================

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import cv2
from tqdm import tqdm
import warnings
import os
warnings.filterwarnings('ignore')


# ============================================================================
# SECTION 2: CONFIGURATION CONSTANTS
# ============================================================================

class Config:
    """Configuration class for VisNet experiments."""
    
    # Device
    DEVICE = torch.device("cuda")
    
    # Experiment parameters
    NUM_TRIALS = 1
    NUM_EPOCHS = 1
    TRAIN_SIZES = [5, 15, 30]
    #TRAIN_SIZES = [5]
    NUM_TEST_SAMPLES = 30

    # Batch sizes
    FEATURE_EXTRACTION_BATCH_SIZE = 1
    BATCH_SIZE = 1
    
    # Network parameters
    LAYER_SIZE = (50, 50)  # 100x100 = 10,000 neurons
    INPUT_SIZE = (32, 32)
    NUM_GABOR_FILTERS = 16
    
    # Sparseness and inhibition
    OUTPUT_SPARSENESS = 0.999
    RECEPTIVE_FIELD_RADIUS1 = 7
    RECEPTIVE_FIELD_RADIUS2 = 7
    
    # Learning parameters
    LEARN_RATE = 0.00001
    LEARN_RATE_L1 = 0.00005
    ALPHA = 0.8
    ALPHA1 = 0.0
    OJA_LEARNING_RATE = 0.0001
    
    # RBF parameters
    WID = 0.5
    
    # Dataset
    DATASET_PATH = "caltech-101/101_ObjectCategories"  # Caltech-101 dataset with 101 classes


# ============================================================================
# SECTION 3: UTILITY FUNCTIONS - FILTERS
# ============================================================================

def gabor_kernel(frequency, theta, sigma=1, lambd=1, gamma=0.5, psi=0):
    """Create a single Gabor filter kernel."""
    sigma_x = sigma
    sigma_y = float(sigma) / gamma
    xmax = max(abs(sigma_x * np.cos(theta)), abs(sigma_y * np.sin(theta)))
    xmax = np.ceil(max(1, xmax))
    ymax = max(abs(sigma_x * np.sin(theta)), abs(sigma_y * np.cos(theta)))
    ymax = np.ceil(max(1, ymax))
    xmin = -xmax
    ymin = -ymax
    (y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)
    gb = np.exp(-0.5 * (x_theta**2 / sigma_x**2 + y_theta**2 / sigma_y**2))
    gb *= np.cos(2 * np.pi / lambd * x_theta + psi)
    return gb


def normalize_kernel(kernel):
    """Normalize kernel to unit L1 norm."""
    return kernel / np.sum(np.abs(kernel))


def create_gabor_filters(frequencies=None, orientations=None, phases=None, fixed_size=21):
    """Generate Gabor filters with fixed size."""
    if frequencies is None:
        frequencies = [0.5, 0.25, 0.125, 0.0625]
    if orientations is None:
        orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    if phases is None:
        phases = [1.0]
    
    filters = []
    for frequency in frequencies:
        for theta in orientations:
            for psi in phases:
                kernel = gabor_kernel(frequency=frequency, theta=theta, psi=psi)
                kernel = normalize_kernel(kernel)
                
                h, w = kernel.shape
                if h < fixed_size or w < fixed_size:
                    pad_h = (fixed_size - h) // 2
                    pad_w = (fixed_size - w) // 2
                    kernel = np.pad(kernel, ((pad_h, fixed_size - h - pad_h), 
                                            (pad_w, fixed_size - w - pad_w)), 
                                   mode='constant', constant_values=0)
                elif h > fixed_size or w > fixed_size:
                    start_h = (h - fixed_size) // 2
                    start_w = (w - fixed_size) // 2
                    kernel = kernel[start_h:start_h+fixed_size, start_w:start_w+fixed_size]
                
                filters.append(kernel)
    return filters


def create_dog_filter(sigma1=1.0, sigma2=1.2, size=7, device='cuda'):
    """Create a Difference of Gaussian (DoG) filter."""
    x = torch.arange(-(size//2), size//2 + 1, dtype=torch.float32, device=device)
    y = torch.arange(-(size//2), size//2 + 1, dtype=torch.float32, device=device)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    r2 = X**2 + Y**2
    g1 = torch.exp(-r2 / (2 * sigma1**2)) / (2 * np.pi * sigma1**2)
    g2 = torch.exp(-r2 / (2 * sigma2**2)) / (2 * np.pi * sigma2**2)
    dog = g1 - 0.6 * g2
    dog = dog / torch.sum(torch.abs(dog))
    return dog.unsqueeze(0).unsqueeze(0)


# ============================================================================
# SECTION 4: UTILITY FUNCTIONS - INHIBITION AND SPARSENESS
# ============================================================================

def apply_local_inhibition_and_sparseness(x, sparseness_target, layer_size, radius, device):
    """
    Apply local inhibition and sparseness to layer activations.
    
    Args:
        x: Input tensor [batch, num_neurons]
        sparseness_target: Target sparseness ratio
        layer_size: Size of spatial layer (100 for 100x100)
        radius: Inhibition radius
        device: Device to run on
    
    Returns:
        Sparse, inhibited activations
    """
    batch_size = x.size(0)
    
    # Reshape to spatial grid
    x_grid = x.view(batch_size, layer_size, layer_size)
    
    # Create inhibition mask
    inhibition_mask = torch.zeros(layer_size, layer_size, device=device)
    for i in range(layer_size):
        for j in range(layer_size):
            for di in range(-radius, radius + 1):
                for dj in range(-radius, radius + 1):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < layer_size and 0 <= nj < layer_size:
                        dist = np.sqrt(di**2 + dj**2)
                        if dist <= radius and dist > 0:
                            inhibition_mask[i, j] += 1.0 / (dist + 1e-8)
    
    # Apply inhibition
    inhibited = x_grid.clone()
    for i in range(batch_size):
        for j in range(layer_size):
            for k in range(layer_size):
                # Apply local inhibition
                neighbors = inhibited[i, 
                                     max(0, j-radius):min(layer_size, j+radius+1),
                                     max(0, k-radius):min(layer_size, k+radius+1)]
                inhibited[i, j, k] = x_grid[i, j, k] - 0.1 * (neighbors.sum() - x_grid[i, j, k])
        
        # Apply sparseness
        values = inhibited[i].flatten()
        num_to_keep = int((1 - sparseness_target) * len(values))
        if num_to_keep > 0:
            _, top_indices = torch.topk(values, num_to_keep)
            mask = torch.zeros_like(values)
            mask[top_indices] = 1.0
            inhibited[i] = (inhibited[i].flatten() * mask).view(layer_size, layer_size)
    
    return inhibited.view(batch_size, -1)


def apply_global_wta_inhibition(x, sparseness_target, device):
    """
    Apply global winner-take-all inhibition.
    
    Args:
        x: Input tensor [batch, num_neurons]
        sparseness_target: Target sparseness ratio
        device: Device to run on
    
    Returns:
        Sparse activations after global WTA
    """
    batch_size = x.size(0)
    
    # Apply global WTA sparseness
    for i in range(batch_size):
        values = x[i]
        num_to_keep = int((1 - sparseness_target) * len(values))
        if num_to_keep > 0:
            _, top_indices = torch.topk(values, num_to_keep)
            mask = torch.zeros_like(values)
            mask[top_indices] = 1.0
            x[i] = x[i] * mask
    
    return x


# ============================================================================
# SECTION 5: MODEL CLASSES - VISNET-LI
# ============================================================================

class SimplifiedVisNetLI(nn.Module):
    """Simplified VisNet-LI for fast feature extraction with Oja's learning."""
    
    def __init__(self, device='cpu'):
        super(SimplifiedVisNetLI, self).__init__()
        self.device = device
        
        # Create Gabor filters
        frequencies = [0.5, 0.25, 0.125, 0.0625]
        orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        phases = [1.0]
        
        gabor_list = []
        for freq in frequencies:
            for theta in orientations:
                for phase in phases:
                    kernel = self._create_gabor_kernel(freq, theta, phase)
                    gabor_list.append(kernel)
        
        self.gabor_filters = torch.stack(gabor_list).unsqueeze(1).to(device)
        
        # Layer sizes
        self.layer_size = Config.LAYER_SIZE[0]
        l1_output = self.layer_size * self.layer_size
        
        # Initialize weights
        self.l1_weights = nn.Parameter(torch.randn(l1_output, 784, device=device) * 0.01)
        self.l2_weights = nn.Parameter(torch.randn(l1_output, 49, device=device) * 0.01)
        self.l3_weights = nn.Parameter(torch.randn(l1_output, 49, device=device) * 0.01)
        self.l4_weights = nn.Parameter(torch.randn(l1_output, 49, device=device) * 0.01)
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(l1_output).to(device)
        self.ln2 = nn.LayerNorm(l1_output).to(device)
        self.ln3 = nn.LayerNorm(l1_output).to(device)
        self.ln4 = nn.LayerNorm(l1_output).to(device)
        
        self.oja_learning_rate = 0.0001
    
    def forward(self, x, return_l4=True, train_mode=False):
        """Forward pass through VisNet-LI."""
        batch_size = x.size(0)
        
        # Apply Gabor filters
        filtered_outputs = []
        for j in range(self.gabor_filters.size(0)):
            filter_j = self.gabor_filters[j:j+1]
            filtered = F.conv2d(x, filter_j, padding='same')
            filtered_outputs.append(filtered)
        
        x = torch.stack(filtered_outputs, dim=1).squeeze(2)
        
        # L1 processing
        x = x.view(batch_size, 32, -1)
        x = x.mean(dim=1)
        
        if x.size(1) < 784:
            x = F.pad(x, (0, 784 - x.size(1)))
        else:
            x = x[:, :784]
        
        x_l1_input = x.clone()
        x = torch.matmul(x.unsqueeze(1), self.l1_weights.t()).squeeze(1)
        x = torch.relu(x)
        x_l1_output = x.clone()
        x = apply_local_inhibition_and_sparseness(x, Config.OUTPUT_SPARSENESS, self.layer_size, 3, self.device)
        x = self.ln1(x)
        
        if train_mode and self.training:
            self.oja_update(self.l1_weights, x_l1_input, x_l1_output, self.oja_learning_rate)
        
        # L2-L4 processing
        x_grid = x.view(batch_size, self.layer_size, self.layer_size)
        
        for layer_idx, (weights, ln) in enumerate([(self.l2_weights, self.ln2),
                                                     (self.l3_weights, self.ln3),
                                                     (self.l4_weights, self.ln4)]):
            x_layer_input = x_grid.clone()
            x = self._apply_spatial_layer(x_grid, weights)
            x = torch.relu(x)
            x_layer_output = x.clone()
            x = apply_local_inhibition_and_sparseness(x, Config.OUTPUT_SPARSENESS, self.layer_size, 3, self.device)
            x = ln(x)
            x_grid = x.view(batch_size, self.layer_size, self.layer_size)
            
            if train_mode and self.training:
                self.oja_update_spatial(weights, x_layer_input, x_layer_output, self.oja_learning_rate)
        
        return x
    
    def _create_gabor_kernel(self, frequency, theta, phase, sigma=None, kernel_size=21):
        """Create a single Gabor filter kernel."""
        if sigma is None:
            sigma = 1.0 / frequency
        
        x = torch.linspace(-kernel_size//2, kernel_size//2, kernel_size)
        y = torch.linspace(-kernel_size//2, kernel_size//2, kernel_size)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        X_theta = X * np.cos(theta) + Y * np.sin(theta)
        Y_theta = -X * np.sin(theta) + Y * np.cos(theta)
        
        gaussian = torch.exp(-(X_theta**2 + Y_theta**2) / (2 * sigma**2))
        sinusoid = torch.cos(2 * np.pi * frequency * X_theta + phase)
        gabor = gaussian * sinusoid
        
        gabor = gabor - gabor.mean()
        gabor = gabor / (gabor.std() + 1e-8)
        
        return gabor
    
    def oja_update(self, weights, x_in, x_out, learning_rate):
        """Oja's learning rule for weight update."""
        with torch.no_grad():
            for i in range(x_in.size(0)):
                y = x_out[i:i+1].t()
                x = x_in[i:i+1]
                
                hebbian = torch.matmul(y, x)
                decay = (y * y) * weights
                
                delta_w = learning_rate * (hebbian - decay)
                weights.add_(delta_w)
            
            weight_norms = torch.norm(weights, dim=1, keepdim=True)
            weights.div_(weight_norms + 1e-8)
    
    def oja_update_spatial(self, weights, x_in, x_out, learning_rate):
        """Oja's learning rule for spatial layers - VECTORIZED for speed."""
        with torch.no_grad():
            patches = F.unfold(x_in.unsqueeze(1), kernel_size=7, padding=3)
            patches = patches.transpose(1, 2)  # [batch, num_patches, patch_size]
            
            hebbian = torch.einsum('bnp,bn->np', patches, x_out)
            
            decay = (x_out ** 2).sum(dim=0, keepdim=True).t() * weights
            
            delta_w = learning_rate * (hebbian - decay)
            weights.add_(delta_w)
            
            weight_norms = torch.norm(weights, dim=1, keepdim=True)
            weights.div_(weight_norms + 1e-8)
    
    def _apply_spatial_layer(self, x_grid, weights):
        """Apply 7x7 spatial receptive field."""
        batch_size = x_grid.size(0)
        
        patches = F.unfold(x_grid.unsqueeze(1), kernel_size=7, padding=3)
        patches = patches.transpose(1, 2)
        
        output = torch.matmul(patches, weights.t())
        output = output.mean(dim=2)
        
        return output


class SimplifiedVisNetLIDoG(nn.Module):
    """VisNet-LI with DoG preprocessing."""
    
    def __init__(self, device='cpu'):
        super(SimplifiedVisNetLIDoG, self).__init__()
        self.device = device
        self.layer_size = Config.LAYER_SIZE[0]
        
        # Create DoG filter
        self.dog_filter = create_dog_filter(device=device)
        
        # Create Gabor filters
        frequencies = [0.0625, 0.125, 0.25, 0.5]
        orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        phases = [1.0]
        
        gabor_list = []
        for freq in frequencies:
            for orient in orientations:
                for phase in phases:
                    kernel = self._create_gabor_kernel(freq, orient, phase)
                    gabor_list.append(kernel)
        
        self.gabor_filters = torch.stack(gabor_list).unsqueeze(1).to(device)
        
        l1_output = self.layer_size * self.layer_size
        
        self.l1_weights = nn.Parameter(torch.randn(l1_output, 784, device=device) * 0.01)
        self.l2_weights = nn.Parameter(torch.randn(l1_output, 49, device=device) * 0.01)
        self.l3_weights = nn.Parameter(torch.randn(l1_output, 49, device=device) * 0.01)
        self.l4_weights = nn.Parameter(torch.randn(l1_output, 49, device=device) * 0.01)
        
        self.ln1 = nn.LayerNorm(l1_output).to(device)
        self.ln2 = nn.LayerNorm(l1_output).to(device)
        self.ln3 = nn.LayerNorm(l1_output).to(device)
        self.ln4 = nn.LayerNorm(l1_output).to(device)
        
        self.oja_learning_rate = 0.0001
    
    def forward(self, x, return_l4=True, train_mode=False):
        """Forward pass with DoG preprocessing."""
        batch_size = x.size(0)
        
        # Apply DoG preprocessing
        if x.dim() == 4 and x.size(1) > 1:
            x = x.mean(dim=1, keepdim=True)
        
        x = F.conv2d(x, self.dog_filter, padding='same')
        
        # Apply Gabor filters
        filtered_outputs = []
        for j in range(self.gabor_filters.size(0)):
            filter_j = self.gabor_filters[j:j+1]
            filtered = F.conv2d(x, filter_j, padding='same')
            filtered_outputs.append(filtered)
        
        x = torch.stack(filtered_outputs, dim=1).squeeze(2)
        
        # L1 processing
        x = x.view(batch_size, 32, -1)
        x = x.mean(dim=1)
        
        if x.size(1) < 784:
            x = F.pad(x, (0, 784 - x.size(1)))
        else:
            x = x[:, :784]
        
        # Manhattan distance activation (L1 norm)
        x_l1_input = x.clone()
        x = self._manhattan_activation(x, self.l1_weights)
        x_l1_output = x.clone()
        x = apply_local_inhibition_and_sparseness(x, Config.OUTPUT_SPARSENESS, self.layer_size, 3, self.device)
        x = self.ln1(x)
        
        if train_mode and self.training:
            self.oja_update(self.l1_weights, x_l1_input, x_l1_output, self.oja_learning_rate)
        
        # L2-L4 processing with Manhattan distance
        x_grid = x.view(batch_size, self.layer_size, self.layer_size)
        
        for layer_idx, (weights, ln) in enumerate([(self.l2_weights, self.ln2),
                                                     (self.l3_weights, self.ln3),
                                                     (self.l4_weights, self.ln4)]):
            x_layer_input = x_grid.clone()
            x = self._apply_spatial_layer_manhattan(x_grid, weights)
            x = apply_local_inhibition_and_sparseness(x, Config.OUTPUT_SPARSENESS, self.layer_size, 3, self.device)
            x = ln(x)
            x_grid = x.view(batch_size, self.layer_size, self.layer_size)
        
        return x
    
    def _manhattan_activation(self, x, weights):
        """Manhattan distance activation (sum of absolute differences)."""
        # x: [batch, input_dim]
        # weights: [output_dim, input_dim]
        
        # Compute Manhattan distance: sum(|x - w|)
        abs_diff = torch.abs(x.unsqueeze(1) - weights.unsqueeze(0))
        manhattan_dist = torch.sum(abs_diff, dim=2)
        
        # Invert distance for activation (smaller distance = higher activation)
        activation = torch.exp(-manhattan_dist)
        
        return activation
    
    def _apply_spatial_layer_manhattan(self, x_grid, weights):
        """Apply 7x7 spatial receptive field with Manhattan distance."""
        batch_size = x_grid.size(0)
        
        patches = F.unfold(x_grid.unsqueeze(1), kernel_size=7, padding=3)
        patches = patches.transpose(1, 2)
        
        # Compute Manhattan distance for each patch
        abs_diff = torch.abs(patches.unsqueeze(2) - weights.unsqueeze(0).unsqueeze(0))
        manhattan_dist = torch.sum(abs_diff, dim=3)
        
        activation = torch.exp(-manhattan_dist)
        output = activation.mean(dim=2)
        
        return output
    
    def _create_gabor_kernel(self, frequency, theta, phase, sigma=None, kernel_size=21):
        """Create a single Gabor filter kernel."""
        if sigma is None:
            sigma = 1.0 / frequency
        
        x = torch.linspace(-kernel_size//2, kernel_size//2, kernel_size)
        y = torch.linspace(-kernel_size//2, kernel_size//2, kernel_size)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        X_theta = X * np.cos(theta) + Y * np.sin(theta)
        Y_theta = -X * np.sin(theta) + Y * np.cos(theta)
        
        gaussian = torch.exp(-(X_theta**2 + Y_theta**2) / (2 * sigma**2))
        sinusoid = torch.cos(2 * np.pi * frequency * X_theta + phase)
        gabor = gaussian * sinusoid
        
        gabor = gabor - gabor.mean()
        gabor = gabor / (gabor.std() + 1e-8)
        
        return gabor
    
    def oja_update(self, weights, x_in, x_out, learning_rate):
        """Oja's learning rule."""
        with torch.no_grad():
            for i in range(x_in.size(0)):
                y = x_out[i:i+1].t()
                x = x_in[i:i+1]
                
                hebbian = torch.matmul(y, x)
                decay = (y * y) * weights
                
                delta_w = learning_rate * (hebbian - decay)
                weights.add_(delta_w)
            
            weight_norms = torch.norm(weights, dim=1, keepdim=True)
            weights.div_(weight_norms + 1e-8)
    
    def oja_update_spatial(self, weights, x_in, x_out, learning_rate):
        """Oja's learning rule for spatial layers."""
        with torch.no_grad():
            batch_size = x_in.size(0)
            
            patches = F.unfold(x_in.unsqueeze(1), kernel_size=7, padding=3)
            patches = patches.transpose(1, 2)
            
            for i in range(x_out.size(0)):
                for j in range(weights.size(0)):
                    x_patch = patches[i, j]
                    y = x_out[i, j].item()
                    
                    hebbian = y * x_patch
                    decay = y * y * weights[j]
                    
                    delta_w = learning_rate * (hebbian - decay)
                    weights[j] += delta_w
            
            weight_norms = torch.norm(weights, dim=1, keepdim=True)
            weights.div_(weight_norms + 1e-8)
    
    def _apply_spatial_layer(self, x_grid, weights):
        """Apply 7x7 spatial receptive field."""
        batch_size = x_grid.size(0)
        
        patches = F.unfold(x_grid.unsqueeze(1), kernel_size=7, padding=3)
        patches = patches.transpose(1, 2)
        
        output = torch.matmul(patches, weights.t())
        output = output.mean(dim=2)
        
        return output


class SimplifiedVisNetLIMD(nn.Module):
    """VisNet-LI with Manhattan Distance learning."""
    
    def __init__(self, device='cpu'):
        super(SimplifiedVisNetLIMD, self).__init__()
        self.device = device
        self.layer_size = Config.LAYER_SIZE[0]
        
        # Create Gabor filters
        frequencies = [0.0625, 0.125, 0.25, 0.5]
        orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        phases = [1.0]
        
        gabor_list = []
        for freq in frequencies:
            for orient in orientations:
                for phase in phases:
                    kernel = self._create_gabor_kernel(freq, orient, phase)
                    gabor_list.append(kernel)
        
        self.gabor_filters = torch.stack(gabor_list).unsqueeze(1).to(device)
        
        l1_output = self.layer_size * self.layer_size
        
        self.l1_weights = nn.Parameter(torch.randn(l1_output, 784, device=device) * 0.01)
        self.l2_weights = nn.Parameter(torch.randn(l1_output, 49, device=device) * 0.01)
        self.l3_weights = nn.Parameter(torch.randn(l1_output, 49, device=device) * 0.01)
        self.l4_weights = nn.Parameter(torch.randn(l1_output, 49, device=device) * 0.01)
        
        self.ln1 = nn.LayerNorm(l1_output).to(device)
        self.ln2 = nn.LayerNorm(l1_output).to(device)
        self.ln3 = nn.LayerNorm(l1_output).to(device)
        self.ln4 = nn.LayerNorm(l1_output).to(device)
        
        self.manhattan_learning_rate = 0.001
    
    def forward(self, x, return_l4=True, train_mode=False):
        """Forward pass with Manhattan distance activation."""
        batch_size = x.size(0)
        
        # Apply Gabor filters
        filtered_outputs = []
        for j in range(self.gabor_filters.size(0)):
            filter_j = self.gabor_filters[j:j+1]
            filtered = F.conv2d(x, filter_j, padding='same')
            filtered_outputs.append(filtered)
        
        x = torch.stack(filtered_outputs, dim=1).squeeze(2)
        
        # L1 processing
        x = x.view(batch_size, 32, -1)
        x = x.mean(dim=1)
        
        if x.size(1) < 784:
            x = F.pad(x, (0, 784 - x.size(1)))
        else:
            x = x[:, :784]
        
        # Manhattan distance activation (L1 norm)
        x_l1_input = x.clone()
        x = self._manhattan_activation(x, self.l1_weights)
        x_l1_output = x.clone()
        x = apply_local_inhibition_and_sparseness(x, Config.OUTPUT_SPARSENESS, self.layer_size, 3, self.device)
        x = self.ln1(x)
        
        if train_mode and self.training:
            self.manhattan_update(self.l1_weights, x_l1_input, x_l1_output, self.manhattan_learning_rate)
        
        # L2-L4 processing with Manhattan distance
        x_grid = x.view(batch_size, self.layer_size, self.layer_size)
        
        for layer_idx, (weights, ln) in enumerate([(self.l2_weights, self.ln2),
                                                     (self.l3_weights, self.ln3),
                                                     (self.l4_weights, self.ln4)]):
            x_layer_input = x_grid.clone()
            x = self._apply_spatial_layer_manhattan(x_grid, weights)
            x_layer_output = x.clone()
            x = apply_local_inhibition_and_sparseness(x, Config.OUTPUT_SPARSENESS, self.layer_size, 3, self.device)
            x = ln(x)
            x_grid = x.view(batch_size, self.layer_size, self.layer_size)
            
            if train_mode and self.training:
                self.manhattan_update_spatial(weights, x_layer_input, x_layer_output, self.manhattan_learning_rate)
        
        return x
    
    def _manhattan_activation(self, x, weights):
        """Manhattan distance activation (sum of absolute differences)."""
        # x: [batch, input_dim]
        # weights: [output_dim, input_dim]
        
        # Compute Manhattan distance: sum(|x - w|)
        abs_diff = torch.abs(x.unsqueeze(1) - weights.unsqueeze(0))
        manhattan_dist = torch.sum(abs_diff, dim=2)
        
        # Invert distance for activation (smaller distance = higher activation)
        activation = torch.exp(-manhattan_dist)
        
        return activation
    
    def _apply_spatial_layer_manhattan(self, x_grid, weights):
        """Apply 7x7 spatial receptive field with Manhattan distance."""
        batch_size = x_grid.size(0)
        
        patches = F.unfold(x_grid.unsqueeze(1), kernel_size=7, padding=3)
        patches = patches.transpose(1, 2)
        
        # Compute Manhattan distance for each patch
        abs_diff = torch.abs(patches.unsqueeze(2) - weights.unsqueeze(0).unsqueeze(0))
        manhattan_dist = torch.sum(abs_diff, dim=3)
        
        activation = torch.exp(-manhattan_dist)
        output = activation.mean(dim=2)
        
        return output
    
    def _create_gabor_kernel(self, frequency, theta, phase, sigma=None, kernel_size=21):
        """Create a single Gabor filter kernel."""
        if sigma is None:
            sigma = 1.0 / frequency
        
        x = torch.linspace(-kernel_size//2, kernel_size//2, kernel_size)
        y = torch.linspace(-kernel_size//2, kernel_size//2, kernel_size)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        X_theta = X * np.cos(theta) + Y * np.sin(theta)
        Y_theta = -X * np.sin(theta) + Y * np.cos(theta)
        
        gaussian = torch.exp(-(X_theta**2 + Y_theta**2) / (2 * sigma**2))
        sinusoid = torch.cos(2 * np.pi * frequency * X_theta + phase)
        gabor = gaussian * sinusoid
        
        gabor = gabor - gabor.mean()
        gabor = gabor / (gabor.std() + 1e-8)
        
        return gabor

    def manhattan_update(self, weights, x_in, x_out, learning_rate):
        """Manhattan distance gradient learning: grad = sign(w - x), dw = lr * (grad - w)."""
        with torch.no_grad():
            # x_in: [batch, input_dim]
            # x_out: [batch, output_dim] - activations
            # weights: [output_dim, input_dim]
            
            for i in range(x_in.size(0)):
                grad = torch.sign(weights - x_in[i:i+1])  # [output_dim, input_dim]
                dw = learning_rate * (grad - weights)
                weights.add_(dw)
            
            # Normalize weights
            weight_norms = torch.norm(weights, dim=1, keepdim=True)
            weights.div_(weight_norms + 1e-8)
    
    def manhattan_update_spatial(self, weights, x_in, x_out, learning_rate):
        """Manhattan distance learning for spatial layers - CHUNKED for speed."""
        with torch.no_grad():
            # Extract 7x7 patches from spatial input
            patches = F.unfold(x_in.unsqueeze(1), kernel_size=7, padding=3)
            patches = patches.transpose(1, 2)  # [batch, num_patches, patch_size]
            
            # Process in chunks to avoid memory issues
            chunk_size = 1000
            num_neurons = weights.size(0)
            
            # Average patches over batch dimension
            avg_patches = patches.mean(dim=0)  # [num_patches, patch_size]
            
            for i in range(0, num_neurons, chunk_size):
                end_i = min(i + chunk_size, num_neurons)
                weights_chunk = weights[i:end_i]  # [chunk_size, patch_size]
                
                # Compute gradient for each patch position
                # weights_chunk: [chunk_size, patch_size]
                # avg_patches: [num_patches, patch_size]
                grad_sum = torch.zeros_like(weights_chunk)
                
                for p in range(avg_patches.size(0)):
                    patch = avg_patches[p:p+1]  # [1, patch_size]
                    grad_sum += torch.sign(weights_chunk - patch)  # [chunk_size, patch_size]
                
                # Average over patches
                grad = grad_sum / avg_patches.size(0)
                
                # Update weights
                dw = learning_rate * (grad - weights_chunk)
                weights[i:end_i].add_(dw)
                
                # Normalize chunk
                weight_norms = torch.norm(weights[i:end_i], dim=1, keepdim=True)
                weights[i:end_i].div_(weight_norms + 1e-8)


class SimplifiedVisNetLIRBF(nn.Module):
    """VisNet-LI with RBF activation."""
    
    def __init__(self, device='cpu', wid=0.5):
        super(SimplifiedVisNetLIRBF, self).__init__()
        self.device = device
        self.layer_size = Config.LAYER_SIZE[0]
        self.wid = wid
        
        # Create Gabor filters
        frequencies = [0.0625, 0.125, 0.25, 0.5]
        orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        phases = [1.0]
        
        gabor_list = []
        for freq in frequencies:
            for orient in orientations:
                for phase in phases:
                    kernel = self._create_gabor_kernel(freq, orient, phase)
                    gabor_list.append(kernel)
        
        self.gabor_filters = torch.stack(gabor_list).unsqueeze(1).to(device)
        
        l1_output = self.layer_size * self.layer_size
        
        self.l1_weights = nn.Parameter(torch.randn(l1_output, 784, device=device) * 0.01)
        self.l2_weights = nn.Parameter(torch.randn(l1_output, 49, device=device) * 0.01)
        self.l3_weights = nn.Parameter(torch.randn(l1_output, 49, device=device) * 0.01)
        self.l4_weights = nn.Parameter(torch.randn(l1_output, 49, device=device) * 0.01)
        
        self.ln1 = nn.LayerNorm(l1_output).to(device)
        self.ln2 = nn.LayerNorm(l1_output).to(device)
        self.ln3 = nn.LayerNorm(l1_output).to(device)
        self.ln4 = nn.LayerNorm(l1_output).to(device)
        
        self.oja_learning_rate = 0.0001
    
    def forward(self, x, return_l4=True, train_mode=False):
        """Forward pass with RBF activation."""
        batch_size = x.size(0)
        
        # Apply Gabor filters
        filtered_outputs = []
        for j in range(self.gabor_filters.size(0)):
            filter_j = self.gabor_filters[j:j+1]
            filtered = F.conv2d(x, filter_j, padding='same')
            filtered_outputs.append(filtered)
        
        x = torch.stack(filtered_outputs, dim=1).squeeze(2)
        
        # L1 processing
        x = x.view(batch_size, 32, -1)
        x = x.mean(dim=1)
        
        if x.size(1) < 784:
            x = F.pad(x, (0, 784 - x.size(1)))
        else:
            x = x[:, :784]
        
        # RBF activation
        x_l1_input = x.clone()
        x = self._rbf_activation(x, self.l1_weights)
        x_l1_output = x.clone()
        x = apply_local_inhibition_and_sparseness(x, Config.OUTPUT_SPARSENESS, self.layer_size, 3, self.device)
        x = self.ln1(x)
        
        if train_mode and self.training:
            self.oja_update(self.l1_weights, x_l1_input, x_l1_output, self.oja_learning_rate)
        
        # L2-L4 processing with RBF
        x_grid = x.view(batch_size, self.layer_size, self.layer_size)
        
        for layer_idx, (weights, ln) in enumerate([(self.l2_weights, self.ln2),
                                                     (self.l3_weights, self.ln3),
                                                     (self.l4_weights, self.ln4)]):
            x_layer_input = x_grid.clone()
            x = self._apply_spatial_layer_rbf(x_grid, weights)
            x_layer_output = x.clone()
            x = apply_local_inhibition_and_sparseness(x, Config.OUTPUT_SPARSENESS, self.layer_size, 3, self.device)
            x = ln(x)
            x_grid = x.view(batch_size, self.layer_size, self.layer_size)
            
            if train_mode and self.training:
                self.oja_update_spatial_rbf(weights, x_layer_input, x_layer_output, self.oja_learning_rate)
        
        return x
    
    def _rbf_activation(self, x, weights):
        """RBF activation using Euclidean distance."""
        # x: [batch, input_dim]
        # weights: [output_dim, input_dim]
        
        # Compute Euclidean distance
        squared_diff = torch.pow(x.unsqueeze(1) - weights.unsqueeze(0), 2)
        euclidean_dist = torch.sqrt(torch.sum(squared_diff, dim=2))
        
        # RBF activation: exp(-wid * distance)
        activation = torch.exp(-self.wid * euclidean_dist)
        
        return activation
    
    def _apply_spatial_layer_rbf(self, x_grid, weights):
        """Apply 7x7 spatial receptive field with RBF."""
        batch_size = x_grid.size(0)
        
        patches = F.unfold(x_grid.unsqueeze(1), kernel_size=7, padding=3)
        patches = patches.transpose(1, 2)
        
        # Compute RBF for each patch
        squared_diff = torch.pow(patches.unsqueeze(2) - weights.unsqueeze(0).unsqueeze(0), 2)
        euclidean_dist = torch.sqrt(torch.sum(squared_diff, dim=3))
        
        activation = torch.exp(-self.wid * euclidean_dist)
        output = activation.mean(dim=2)
        
        return output
    
    def oja_update(self, weights, x_in, x_out, learning_rate):
        """Oja's learning rule."""
        with torch.no_grad():
            for i in range(x_in.size(0)):
                y = x_out[i:i+1].t()
                x = x_in[i:i+1]
                
                hebbian = torch.matmul(y, x)
                decay = (y * y) * weights
                
                delta_w = learning_rate * (hebbian - decay)
                weights.add_(delta_w)
            
            weight_norms = torch.norm(weights, dim=1, keepdim=True)
            weights.div_(weight_norms + 1e-8)
    
    def oja_update_spatial_rbf(self, weights, x_in, x_out, learning_rate):
        """Oja's learning rule for spatial layers with RBF."""
        with torch.no_grad():
            batch_size = x_in.size(0)
            
            patches = F.unfold(x_in.unsqueeze(1), kernel_size=7, padding=3)
            patches = patches.transpose(1, 2)
            
            for i in range(x_out.size(0)):
                for j in range(weights.size(0)):
                    x_patch = patches[i, j]
                    y = x_out[i, j].item()
                    
                    hebbian = y * x_patch
                    decay = y * y * weights[j]
                    
                    delta_w = learning_rate * (hebbian - decay)
                    weights[j] += delta_w
            
            weight_norms = torch.norm(weights, dim=1, keepdim=True)
            weights.div_(weight_norms + 1e-8)
    
    def _create_gabor_kernel(self, frequency, theta, phase, sigma=None, kernel_size=21):
        """Create a single Gabor filter kernel."""
        if sigma is None:
            sigma = 1.0 / frequency
        
        x = torch.linspace(-kernel_size//2, kernel_size//2, kernel_size)
        y = torch.linspace(-kernel_size//2, kernel_size//2, kernel_size)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        X_theta = X * np.cos(theta) + Y * np.sin(theta)
        Y_theta = -X * np.sin(theta) + Y * np.cos(theta)
        
        gaussian = torch.exp(-(X_theta**2 + Y_theta**2) / (2 * sigma**2))
        sinusoid = torch.cos(2 * np.pi * frequency * X_theta + phase)
        gabor = gaussian * sinusoid
        
        gabor = gabor - gabor.mean()
        gabor = gabor / (gabor.std() + 1e-8)
        
        return gabor


class SimplifiedVisNet(nn.Module):
    """Simplified VisNet with Global WTA Inhibition."""
    
    def __init__(self, device='cpu'):
        super(SimplifiedVisNet, self).__init__()
        self.device = device
        self.layer_size = Config.LAYER_SIZE[0]
        
        # Create Gabor filters
        frequencies = [0.0625, 0.125, 0.25, 0.5]
        orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        phases = [1.0]
        
        gabor_list = []
        for freq in frequencies:
            for orient in orientations:
                for phase in phases:
                    kernel = self._create_gabor_kernel(freq, orient, phase)
                    gabor_list.append(kernel)
        
        self.gabor_filters = torch.stack(gabor_list).unsqueeze(1).to(device)
        
        l1_output = self.layer_size * self.layer_size
        
        self.l1_weights = nn.Parameter(torch.randn(l1_output, 784, device=device) * 0.01)
        self.l2_weights = nn.Parameter(torch.randn(l1_output, 49, device=device) * 0.01)
        self.l3_weights = nn.Parameter(torch.randn(l1_output, 49, device=device) * 0.01)
        self.l4_weights = nn.Parameter(torch.randn(l1_output, 49, device=device) * 0.01)
        
        self.ln1 = nn.LayerNorm(l1_output).to(device)
        self.ln2 = nn.LayerNorm(l1_output).to(device)
        self.ln3 = nn.LayerNorm(l1_output).to(device)
        self.ln4 = nn.LayerNorm(l1_output).to(device)
        
        self.oja_learning_rate = 0.0001
    
    def forward(self, x, return_l4=True, train_mode=False):
        """Forward pass with global WTA."""
        batch_size = x.size(0)
        
        # Apply Gabor filters
        filtered_outputs = []
        for j in range(self.gabor_filters.size(0)):
            filter_j = self.gabor_filters[j:j+1]
            filtered = F.conv2d(x, filter_j, padding='same')
            filtered_outputs.append(filtered)
        
        x = torch.stack(filtered_outputs, dim=1).squeeze(2)
        
        # L1 processing
        x = x.view(batch_size, 32, -1)
        x = x.mean(dim=1)
        
        if x.size(1) < 784:
            x = F.pad(x, (0, 784 - x.size(1)))
        else:
            x = x[:, :784]
        
        x_l1_input = x.clone()
        x = torch.matmul(x.unsqueeze(1), self.l1_weights.t()).squeeze(1)
        x = torch.relu(x)
        x_l1_output = x.clone()
        x = apply_global_wta_inhibition(x, Config.OUTPUT_SPARSENESS, self.device)
        x = self.ln1(x)
        
        if train_mode and self.training:
            self.oja_update(self.l1_weights, x_l1_input, x_l1_output, self.oja_learning_rate)
        
        # L2-L4 processing
        x_grid = x.view(batch_size, self.layer_size, self.layer_size)
        
        for layer_idx, (weights, ln) in enumerate([(self.l2_weights, self.ln2),
                                                     (self.l3_weights, self.ln3),
                                                     (self.l4_weights, self.ln4)]):
            x_layer_input = x_grid.clone()
            x = self._apply_spatial_layer(x_grid, weights)
            x = torch.relu(x)
            x_layer_output = x.clone()
            x = apply_global_wta_inhibition(x, Config.OUTPUT_SPARSENESS, self.device)
            x = ln(x)
            x_grid = x.view(batch_size, self.layer_size, self.layer_size)
            
            if train_mode and self.training:
                self.oja_update_spatial(weights, x_layer_input, x_layer_output, self.oja_learning_rate)
        
        return x
    
    def _create_gabor_kernel(self, frequency, theta, phase, sigma=None, kernel_size=21):
        """Create a single Gabor filter kernel."""
        if sigma is None:
            sigma = 1.0 / frequency
        
        x = torch.linspace(-kernel_size//2, kernel_size//2, kernel_size)
        y = torch.linspace(-kernel_size//2, kernel_size//2, kernel_size)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        X_theta = X * np.cos(theta) + Y * np.sin(theta)
        Y_theta = -X * np.sin(theta) + Y * np.cos(theta)
        
        gaussian = torch.exp(-(X_theta**2 + Y_theta**2) / (2 * sigma**2))
        sinusoid = torch.cos(2 * np.pi * frequency * X_theta + phase)
        gabor = gaussian * sinusoid
        
        gabor = gabor - gabor.mean()
        gabor = gabor / (gabor.std() + 1e-8)
        
        return gabor
    
    def oja_update(self, weights, x_in, x_out, learning_rate):
        """Oja's learning rule."""
        with torch.no_grad():
            for i in range(x_in.size(0)):
                y = x_out[i:i+1].t()
                x = x_in[i:i+1]
                
                hebbian = torch.matmul(y, x)
                decay = (y * y) * weights
                
                delta_w = learning_rate * (hebbian - decay)
                weights.add_(delta_w)
            
            weight_norms = torch.norm(weights, dim=1, keepdim=True)
            weights.div_(weight_norms + 1e-8)
    
    def oja_update_spatial(self, weights, x_in, x_out, learning_rate):
        """Oja's learning rule for spatial layers - VECTORIZED for speed."""
        with torch.no_grad():
            patches = F.unfold(x_in.unsqueeze(1), kernel_size=7, padding=3)
            patches = patches.transpose(1, 2)  # [batch, num_patches, patch_size]
            
            hebbian = torch.einsum('bnp,bn->np', patches, x_out)
            
            decay = (x_out ** 2).sum(dim=0, keepdim=True).t() * weights
            
            delta_w = learning_rate * (hebbian - decay)
            weights.add_(delta_w)
            
            weight_norms = torch.norm(weights, dim=1, keepdim=True)
            weights.div_(weight_norms + 1e-8)
    
    def _apply_spatial_layer(self, x_grid, weights):
        """Apply 7x7 spatial receptive field."""
        batch_size = x_grid.size(0)
        
        patches = F.unfold(x_grid.unsqueeze(1), kernel_size=7, padding=3)
        patches = patches.transpose(1, 2)
        
        output = torch.matmul(patches, weights.t())
        output = output.mean(dim=2)
        
        return output


# ============================================================================
# SECTION 6: MODEL SETUP FUNCTIONS
# ============================================================================

def setup_visnet_li_architecture(device):
    """Initialize VisNet-LI architecture."""
    print("Initializing VisNet-LI architecture...")
    model = SimplifiedVisNetLI(device=device).to(device)
    model.eval()
    return model


def setup_visnet_li_dog_architecture(device):
    """Initialize VisNet-LI-DoG architecture."""
    print("Initializing VisNet-LI-DoG architecture...")
    model = SimplifiedVisNetLIDoG(device=device).to(device)
    model.eval()
    return model


def setup_simplified_visnet_architecture(device):
    """Initialize Simplified VisNet architecture."""
    print("Initializing Simplified VisNet architecture...")
    model = SimplifiedVisNet(device=device).to(device)
    model.eval()
    return model


def setup_visnet_md_architecture(device):
    """Initialize VisNet-MD-Linear architecture."""
    print("Initializing VisNet-MD-Linear architecture...")
    model = SimplifiedVisNetLIMD(device=device).to(device)
    model.eval()
    return model


def setup_visnet_rbf_architecture(device):
    """Initialize VisNet-RBF-MD architecture."""
    print("Initializing VisNet-RBF-MD architecture...")
    model = SimplifiedVisNetLIRBF(device=device, wid=Config.WID).to(device)
    model.eval()
    return model


# ============================================================================
# SECTION 7: FEATURE EXTRACTION FUNCTIONS
# ============================================================================

def extract_features_visnet_li(image, model, device):
    """Extract L4 features using VisNet-LI."""
    with torch.no_grad():
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        
        image = image.to(device)
        
        if image.size(1) == 3:
            x = 0.2989 * image[:, 0] + 0.5870 * image[:, 1] + 0.1140 * image[:, 2]
            x = x.unsqueeze(1)
        elif image.size(1) == 1:
            x = image
        else:
            raise ValueError(f"Unexpected number of channels: {image.size(1)}")
        
        x = model.forward(x, return_l4=True)
        return x.cpu().numpy().flatten()


def update_weights_oja_learning(model, image, learning_rate=0.0001):
    """Update weights using Oja's Hebbian learning rule."""
    if not isinstance(image, torch.Tensor):
        img_tensor = torch.from_numpy(np.array(image)).float()
    else:
        img_tensor = image.float()
    
    if img_tensor.dim() == 2:
        img_tensor = img_tensor.unsqueeze(0)
    
    model_device = next(model.parameters()).device
    img_tensor = img_tensor.to(model_device)
    
    model.train()
    _ = model.forward(img_tensor, return_l4=True, train_mode=True)


# ============================================================================
# SECTION 8: EXPERIMENT RUNNER
# ============================================================================

def run_single_experiment_with_unsupervised_learning(train_size_per_class, method_name="VisNet-LI", dataset=None):
    """
    Run a single experiment with unsupervised learning.
    
    Args:
        train_size_per_class: Number of training samples per class
        method_name: Name of the method to use
        dataset: Dataset to use
    
    Returns:
        Dictionary with experiment results
    """
    if dataset is None:
        print("Error: No dataset provided")
        return None
    
    results = []
    
    for trial in range(Config.NUM_TRIALS):
        print(f"\n--- Trial {trial+1}/{Config.NUM_TRIALS} ---")
        
        # Initialize model for this trial
        if method_name == "VisNet-LI":
            model = setup_visnet_li_architecture(Config.DEVICE)
        elif method_name == "Simplified VisNet":
            model = setup_simplified_visnet_architecture(Config.DEVICE)
        elif method_name == "VisNet-MD-Linear":
            model = setup_visnet_md_architecture(Config.DEVICE)
        elif method_name == "VisNet-RBF-MD":
            model = setup_visnet_rbf_architecture(Config.DEVICE)
        else:
            print(f"Unknown method: {method_name}")
            continue
        
        # Split data
        unique_labels = np.unique([label for _, label in dataset])
        
        if len(unique_labels) < 2:
            print(f"Trial {trial+1}: Need at least 2 classes, found {len(unique_labels)}")
            continue
        
        class_0_samples = [(img, label) for img, label in dataset if label == unique_labels[0]]
        class_1_samples = [(img, label) for img, label in dataset if label == unique_labels[1]]
        
        np.random.seed(trial*42 + 10)
        np.random.shuffle(class_0_samples)
        np.random.shuffle(class_1_samples)
        
        train_samples = class_0_samples[:train_size_per_class] + class_1_samples[:train_size_per_class]
        test_samples = class_0_samples[train_size_per_class:train_size_per_class + Config.NUM_TEST_SAMPLES] + \
                      class_1_samples[train_size_per_class:train_size_per_class + Config.NUM_TEST_SAMPLES]
        
        if len(test_samples) < 10:
            print(f"Trial {trial+1}: Not enough test samples")
            continue
        
        print(f"    Train samples: {len(train_samples)}, Test samples: {len(test_samples)}")
        
        # Unsupervised learning
        print("    Phase 1: Unsupervised learning...")
        for epoch in range(Config.NUM_EPOCHS):
            for img, label in tqdm(train_samples, desc=f"Epoch {epoch+1}/{Config.NUM_EPOCHS}"):
                update_weights_oja_learning(model, img)
        
        # Extract features
        train_features = []
        for img, label in train_samples:
            feat = extract_features_visnet_li(img, model, Config.DEVICE)
            train_features.append(feat)
        
        test_features = []
        for img, label in test_samples:
            feat = extract_features_visnet_li(img, model, Config.DEVICE)
            test_features.append(feat)
        
        # Prepare labels
        y_train = [label for _, label in train_samples]
        y_test = [label for _, label in test_samples]
        
        # Train SVM
        svm = LinearSVC(max_iter=10000, random_state=trial*42 + 10, dual=False)
        svm.fit(train_features, y_train)
        
        # Test
        y_pred = svm.predict(test_features)
        accuracy = accuracy_score(y_test, y_pred)
        
        results.append(accuracy)
        print(f"Trial {trial+1} accuracy: {accuracy*100:.2f}%")
    
    return {
        'mean': np.mean(results),
        'std': np.std(results),
        'all_results': results
    }


# ============================================================================
# SECTION 9: MAIN EXECUTION
# ============================================================================

def plot_results(all_results, save_path="visnet_results.png"):
    """Plot comparison results."""
    plt.figure(figsize=(10, 6))
    
    colors = {
        'VisNet-LI': '#F39C12',
        'VisNet-LI-DoG': '#E74C3C',
        'Simplified VisNet': '#000000',
        'VisNet-MD-Linear': '#9B59B6',
        'VisNet-RBF-MD': '#2ECC71'
    }
    
    markers = {
        'VisNet-LI': 'd',
        'VisNet-LI-DoG': 's',
        'Simplified VisNet': 'p',
        'VisNet-MD-Linear': 'v',
        'VisNet-RBF-MD': '^'
    }
    
    for method, results in all_results.items():
        train_sizes = sorted(results.keys())
        means = [results[size]['mean'] * 100 for size in train_sizes]
        stds = [results[size]['std'] * 100 for size in train_sizes]
        
        color = colors.get(method, '#000000')
        marker = markers.get(method, 'o')
        
        plt.errorbar(train_sizes, means, yerr=stds, 
                     label=method, color=color, marker=marker, 
                     linewidth=2, markersize=8, capsize=5)
    
    plt.xlabel('Training Samples Per Class', fontsize=12)
    plt.ylabel('Classification Accuracy (%)', fontsize=12)
    plt.title('VisNet Variants Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim(40, 100)
    plt.xlim(0, 35)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved: {save_path}")
    plt.show()


def main():
    """Main execution function."""
    print("="*80)
    print("VISNET EXPERIMENT RUNNER")
    print("="*80)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available. Falling back to CPU.")
        Config.DEVICE = torch.device("cpu")
    else:
        print(f"CUDA is available. Using device: {torch.cuda.get_device_name(0)}")
    
    print(f"\nDevice: {Config.DEVICE}")
    print(f"Trials: {Config.NUM_TRIALS}")
    print(f"Epochs: {Config.NUM_EPOCHS}")
    print(f"Train sizes: {Config.TRAIN_SIZES}")
    
    # Load dataset
    print("\nLoading dataset...")
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Use Caltech-101 if available
    if os.path.exists(Config.DATASET_PATH):
        dataset = datasets.ImageFolder(Config.DATASET_PATH, transform=transform)
        print(f"Loaded dataset with {len(dataset)} samples")
        
        # Check if we have at least 2 classes
        unique_labels = np.unique([label for _, label in dataset])
        print(f"Found {len(unique_labels)} classes in dataset")
        
        if len(unique_labels) < 2:
            print(f"ERROR: Need at least 2 classes for binary classification!")
            print(f"Please check your dataset at: {Config.DATASET_PATH}")
            print(f"\nDataset structure should be:")
            print(f"  {Config.DATASET_PATH}/")
            print(f"    class1/")
            print(f"      image1.jpg")
            print(f"      image2.jpg")
            print(f"    class2/")
            print(f"      image1.jpg")
            print(f"      image2.jpg")
            return
    else:
        print(f"Dataset not found at {Config.DATASET_PATH}")
        print("Please provide your own dataset")
        return
    
    # Run experiments
    methods = [
        "Simplified VisNet",
        "VisNet-LI", 
        "VisNet-MD",
        "VisNet-RBF"
    ]
    all_results = {}
    
    for method in methods:
        print(f"\n{'='*80}")
        print(f"METHOD: {method.upper()}")
        print(f"{'='*80}")
        
        method_results = {}
        
        for train_size in Config.TRAIN_SIZES:
            print(f"\n--- Training Size: {train_size} samples per class ---")
            
            result = run_single_experiment_with_unsupervised_learning(
                train_size_per_class=train_size,
                method_name=method,
                dataset=dataset
            )
            
            if result:
                method_results[train_size] = result
                print(f"  Mean accuracy: {result['mean']*100:.2f}% ± {result['std']*100:.2f}%")
        
        all_results[method] = method_results
    
    # Print summary
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    
    for method, results in all_results.items():
        print(f"\n{method}:")
        for train_size, result in results.items():
            print(f"  {train_size} samples: {result['mean']*100:.2f}% ± {result['std']*100:.2f}%")
    
    # Plot results
    plot_results(all_results)


if __name__ == "__main__":
    main()

