"""
VisNet: Fully Functional & Optimized Implementation
===================================================
Variants:
 - VisNet-LI          (Oja learning + local inhibition)
 - Simplified VisNet   (Global WTA)
 - VisNet-MD-Linear   (Manhattan distance competitive Hebbian)
 - VisNet-RBF         (RBF competitive Hebbian)

Optimizations:
 - Vectorized Oja/competitive updates
 - Fast local/global inhibition (conv2d + topk)
 - Mini‑batch unsupervised training
 - Random init noise to break symmetry
"""

# ============================================================================
# IMPORTS
# ============================================================================

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt, os, warnings
warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TRAIN_SIZES = [5, 15, 30]
    NUM_EPOCHS = 3
    BATCH_SIZE = 8                   # mini‑batch size
    NUM_TEST_SAMPLES = 30
    NUM_TRIALS = 3                   # Number of trials per experiment (for statistics)
    LAYER_SIZE = 100
    INPUT_SIZE = (32, 32)
    OUTPUT_SPARSENESS = 0.99         # keep top 1 % active
    RECEPTIVE_FIELD_RADIUS = 3
    OJA_LR = 5e-4                    # tuned LR for Oja learning
    MD_LR = 5e-4                     # Manhattan distance learning rate (same as Oja for fair comparison)
    RBF_LR = 5e-4                    # RBF learning rate (higher for competitive learning, 0.05 for fast convergence)
    WID = 0.5                        # RBF width (higher = broader tuning)
    DATASET_PATH = None  # Will be resolved below

# ============================================================================
# RECEPTIVE FIELD MASKS
# ============================================================================

def create_circular_mask(h, w, center=None, radius=None):
    """Create a circular mask for receptive fields"""
    if center is None:
        center = (int(w/2), int(h/2))
    if radius is None:
        radius = min(center[0], center[1], w-center[0], h-center[1])
    
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    mask = dist_from_center <= radius
    return torch.from_numpy(mask).float()

def create_l1_receptive_field_mask(layer_size, input_size, num_gabor_filters, radius, device):
    """
    Create L1 receptive field masks: 7×7 spatial × all Gabor filters for each output neuron
    Returns mask of shape [layer_size², num_gabor_filters * input_size] 
    This matches the reference: mask is applied to full Gabor tensor (all filters × spatial)
    """
    h, w = input_size
    # Create spatial mask [layer_size, layer_size, h, w]
    mask_2d = torch.zeros(layer_size, layer_size, h, w, device=device)
    
    # Calculate spatial step for output neurons over input
    step_h = h / layer_size
    step_w = w / layer_size
    
    for i in range(layer_size):
        for j in range(layer_size):
            center_h = int((i + 0.5) * step_h)
            center_w = int((j + 0.5) * step_w)
            mask_2d[i, j, :, :] = create_circular_mask(h, w, center=(center_w, center_h), radius=radius).to(device)
    
    # Expand to include all Gabor filters: [layer_size², num_gabor_filters, h, w]
    mask_2d_expanded = mask_2d.view(layer_size * layer_size, 1, h, w).repeat(1, num_gabor_filters, 1, 1)
    # Flatten to [layer_size², num_gabor_filters * h * w]
    mask_flat = mask_2d_expanded.view(layer_size * layer_size, num_gabor_filters * h * w)
    return mask_flat

# ============================================================================
# FILTER UTILITIES
# ============================================================================

def create_gabor_kernel(frequency, theta, phase=0, sigma=None, kernel_size=21):
    if sigma is None: sigma = 1.0 / frequency
    x = torch.linspace(-kernel_size//2, kernel_size//2, kernel_size)
    y = torch.linspace(-kernel_size//2, kernel_size//2, kernel_size)
    X,Y = torch.meshgrid(x,y, indexing='ij')
    X_t = X*np.cos(theta)+Y*np.sin(theta)
    g = torch.exp(-(X_t**2 + Y**2)/(2*sigma**2))
    g *= torch.cos(2*np.pi*frequency*X_t + phase)
    g -= g.mean(); g /= g.std()+1e-8
    return g

def gabor_bank():
    freqs=[0.0625,0.125,0.25,0.5]; orients=[0,np.pi/4,np.pi/2,3*np.pi/4]
    return torch.stack([create_gabor_kernel(f,o) for f in freqs for o in orients]).unsqueeze(1)

def create_dog_filter(s1=1.0,s2=1.2,size=7,device="cpu"):
    x=torch.arange(-(size//2),size//2+1,device=device,dtype=torch.float32)
    X,Y=torch.meshgrid(x,x, indexing='ij'); r2=X**2+Y**2
    g1=torch.exp(-r2/(2*s1**2))/(2*np.pi*s1**2)
    g2=torch.exp(-r2/(2*s2**2))/(2*np.pi*s2**2)
    dog=g1-0.6*g2; dog/=dog.abs().sum()
    return dog.unsqueeze(0).unsqueeze(0)

def create_dog_filter_rgb(sigma1=1.0, sigma2=1.2, size=3, device='cpu'):
    """Create a Difference of Gaussian (DoG) filter for RGB processing."""
    x = torch.arange(-(size//2), size//2 + 1, dtype=torch.float32, device=device)
    y = torch.arange(-(size//2), size//2 + 1, dtype=torch.float32, device=device)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    r2 = X**2 + Y**2
    g1 = torch.exp(-r2 / (2 * sigma1**2)) / (2 * np.pi * sigma1**2)
    g2 = torch.exp(-r2 / (2 * sigma2**2)) / (2 * np.pi * sigma2**2)
    dog = g1 - 0.6 * g2
    dog = dog / torch.sum(torch.abs(dog))
    return dog.unsqueeze(0).unsqueeze(0)

def apply_dog_filter_rgb(image, dog_filter):
    """Apply DoG filter to RGB image with color opponency channels."""
    R = image[:, 0:1]
    G = image[:, 1:2]
    B = image[:, 2:3]
    
    luminance = (R + G + B) / 3.0
    R_G = R - G
    B_G = B - G
    
    luminance_filtered = F.conv2d(luminance, dog_filter, padding='same')
    R_G_filtered = F.conv2d(R_G, dog_filter, padding='same')
    B_G_filtered = F.conv2d(B_G, dog_filter, padding='same')
    
    luminance_filtered = (luminance_filtered - luminance_filtered.min()) / (luminance_filtered.max() - luminance_filtered.min() + 1e-8)
    R_G_filtered = (R_G_filtered - R_G_filtered.min()) / (R_G_filtered.max() - R_G_filtered.min() + 1e-8)
    B_G_filtered = (B_G_filtered - B_G_filtered.min()) / (B_G_filtered.max() - B_G_filtered.min() + 1e-8)
    
    filtered = torch.cat([luminance_filtered, R_G_filtered, B_G_filtered], dim=1)
    return filtered

# ============================================================================
# INHIBITION & SPARSENESS
# ============================================================================

def apply_local_inhibition_and_sparseness(x,sparsity,layer_radius,layer_size,device):
    B=x.size(0)
    xg=x.view(B,1,layer_size,layer_size)
    yy,xx=torch.meshgrid(
        torch.arange(-layer_radius,layer_radius+1,device=device),
        torch.arange(-layer_radius,layer_radius+1,device=device), indexing='ij')
    dist=torch.sqrt(xx**2+yy**2)
    k=torch.exp(-0.5*(dist/layer_radius)**2); k/=k.sum()
    inhibited=xg-0.1*(F.conv2d(xg,k.unsqueeze(0).unsqueeze(0),padding=layer_radius)-xg)
    inhibited=inhibited.view(B,-1)
    keep=int((1-sparsity)*inhibited.size(1))
    if keep>0:
        vals,_=torch.topk(inhibited,keep,dim=1)
        thr=vals[:,-1].unsqueeze(1)
        inhibited=torch.where(inhibited>=thr,inhibited,torch.zeros_like(inhibited))
    return inhibited

def apply_global_wta_inhibition(x,sparsity):
    B,N=x.shape
    keep=int((1-sparsity)*N)
    if keep<=0:return torch.zeros_like(x)
    vals,idx=torch.topk(x,keep,dim=1)
    mask=torch.zeros_like(x).scatter_(1,idx,1.0)
    return x*mask

# ============================================================================
# LEARNING RULE: OJA
# ============================================================================

def oja_update(w,x_in,x_out,lr):
    with torch.no_grad():
        heb=torch.matmul(x_out.t(),x_in)/x_in.size(0)
        dec=(x_out.pow(2).mean(dim=0,keepdim=True).t())*w
        w.add_(lr*(heb-dec))
        w.div_(torch.norm(w,dim=1,keepdim=True)+1e-8)

# ============================================================================
# MODELS
# ============================================================================

class BaseVisNet(nn.Module):
    def __init__(self,device,num_input_channels=1):
        super().__init__()
        self.device=device
        self.filters=gabor_bank().to(device)
        self.layer_size=Config.LAYER_SIZE
        self.num_input_channels=num_input_channels
        
        # Total number of filters: 16 Gabor × num_input_channels (1 for gray, 3 for RGB DoG)
        num_gabor_filters = self.filters.size(0)  # 16 filters
        num_total_filters = num_gabor_filters * num_input_channels  # 16 for gray, 48 for RGB DoG
        
        inp = num_total_filters * Config.INPUT_SIZE[0] * Config.INPUT_SIZE[1]
        o=self.layer_size**2
        self.l1=nn.Parameter(torch.randn(o,inp,device=device)*0.01)
        with torch.no_grad(): self.l1 += 0.05*torch.randn_like(self.l1)
        self.ln=nn.LayerNorm(o)
        
        # Create L1 receptive field mask (7×7 spatial × all filters for each output neuron)
        self.l1_mask = create_l1_receptive_field_mask(
            self.layer_size, Config.INPUT_SIZE, num_total_filters, Config.RECEPTIVE_FIELD_RADIUS, device
        )

class SimplifiedVisNetLI(BaseVisNet):
    def forward(self,x,train_mode=False):
        B=x.size(0)
        # Apply Gabor filters: [B, num_filters, H, W]
        f=[F.conv2d(x,g.unsqueeze(0),padding='same') for g in self.filters]
        gabor_tensor = torch.stack(f,dim=1)  # [B, num_filters, H, W]
        
        # Reshape to [B, num_filters * H * W] - keep all filters (don't average yet)
        num_filters = gabor_tensor.size(1)
        h, w = gabor_tensor.size(2), gabor_tensor.size(3)
        # Use -1 to auto-calculate flattened dimension (handles any spatial size)
        gabor_flat = gabor_tensor.view(B, -1)  # [B, num_filters * H * W]
        
        # Apply receptive field mask to full Gabor tensor (all filters × spatial)
        gabor_expanded = gabor_flat.unsqueeze(1)  # [B, 1, num_filters * H * W]
        gabor_masked = gabor_expanded * self.l1_mask.unsqueeze(0)  # [B, layer_size², num_filters * H * W]
        
        # Now compute activations: y[i] = masked_gabor[i] @ w[i]
        # self.l1 weights should be [layer_size², num_filters * H * W] to match mask
        y = torch.relu((gabor_masked * self.l1.unsqueeze(0)).sum(dim=2))  # [B, layer_size²]
        y=apply_local_inhibition_and_sparseness(y,Config.OUTPUT_SPARSENESS,
                    Config.RECEPTIVE_FIELD_RADIUS,self.layer_size,self.device)
        y=self.ln(y)
        if train_mode and self.training:
            with torch.no_grad():
                # Vectorized Oja update with masked Gabor inputs
                heb = (y.unsqueeze(2) * gabor_masked).mean(0)  # [layer_size², num_filters * H * W]
                dec = (y.pow(2).mean(0).unsqueeze(1) * self.l1)  # [layer_size², num_filters * H * W]
                self.l1.add_(Config.OJA_LR * (heb - dec))
                self.l1.div_(torch.norm(self.l1, dim=1, keepdim=True) + 1e-8)
        return y

class SimplifiedVisNetLIDoGRGB(BaseVisNet):
    """VisNet-LI with RGB DoG preprocessing (color opponency channels)
    Uses 3 DoG channels × 16 Gabor filters = 48 feature channels"""
    def __init__(self,device):
        super().__init__(device, num_input_channels=3)  # 3 channels for RGB DoG
        self.dog_rgb=create_dog_filter_rgb(device=device)
    
    def forward(self,x,train_mode=False):
        B=x.size(0)
        # Apply RGB DoG filter (creates 3 channels: luminance, R-G, B-G)
        if x.size(1) == 3:
            x = apply_dog_filter_rgb(x, self.dog_rgb)  # [B, 3, H, W]
        elif x.size(1) == 1:
            # If grayscale, replicate to 3 channels
            x = x.repeat(1, 3, 1, 1)
        
        # Apply Gabor filters to EACH of the 3 DoG channels
        # This gives us 3 × 16 = 48 Gabor responses
        gabor_outputs = []
        for c in range(x.size(1)):  # For each DoG channel (3 channels)
            x_channel = x[:, c:c+1, :, :]  # [B, 1, H, W]
            for g in self.filters:  # For each Gabor filter (16 filters)
                gabor_out = F.conv2d(x_channel, g.unsqueeze(0), padding='same')
                gabor_outputs.append(gabor_out)
        
        # Stack all 48 Gabor outputs: [B, 48, H, W]
        gabor_tensor = torch.stack(gabor_outputs, dim=1)  # [B, 48, H, W]
        
        # Reshape to [B, 48 * H * W]
        num_filters = gabor_tensor.size(1)
        h, w = gabor_tensor.size(2), gabor_tensor.size(3)
        gabor_flat = gabor_tensor.view(B, -1)  # [B, 48 * H * W]
        
        # Apply receptive field mask to full Gabor tensor (all filters × spatial)
        gabor_expanded = gabor_flat.unsqueeze(1)  # [B, 1, 48 * H * W]
        gabor_masked = gabor_expanded * self.l1_mask.unsqueeze(0)  # [B, layer_size², 48 * H * W]
        
        # ReLU activation + Local inhibition
        y = torch.relu((gabor_masked * self.l1.unsqueeze(0)).sum(dim=2))  # [B, layer_size²]
        y=apply_local_inhibition_and_sparseness(y,Config.OUTPUT_SPARSENESS,
                    Config.RECEPTIVE_FIELD_RADIUS,self.layer_size,self.device)
        y=self.ln(y)
        
        if train_mode and self.training:
            with torch.no_grad():
                # Vectorized Oja update with masked Gabor inputs
                heb = (y.unsqueeze(2) * gabor_masked).mean(0)  # [layer_size², 48 * H * W]
                dec = (y.pow(2).mean(0).unsqueeze(1) * self.l1)  # [layer_size², 48 * H * W]
                self.l1.add_(Config.OJA_LR * (heb - dec))
                self.l1.div_(torch.norm(self.l1, dim=1, keepdim=True) + 1e-8)
        return y

class SimplifiedVisNet(BaseVisNet):
    def forward(self,x,train_mode=False):
        B=x.size(0)
        # Apply Gabor filters: [B, num_filters, H, W]
        f=[F.conv2d(x,g.unsqueeze(0),padding='same') for g in self.filters]
        gabor_tensor = torch.stack(f,dim=1)  # [B, num_filters, H, W]
        
        # Reshape to [B, num_filters * H * W] - keep all filters (don't average yet)
        num_filters = gabor_tensor.size(1)
        h, w = gabor_tensor.size(2), gabor_tensor.size(3)
        # Use -1 to auto-calculate flattened dimension (handles any spatial size)
        gabor_flat = gabor_tensor.view(B, -1)  # [B, num_filters * H * W]
        
        # Apply receptive field mask to full Gabor tensor (all filters × spatial)
        gabor_expanded = gabor_flat.unsqueeze(1)  # [B, 1, num_filters * H * W]
        gabor_masked = gabor_expanded * self.l1_mask.unsqueeze(0)  # [B, layer_size², num_filters * H * W]
        
        y = torch.relu((gabor_masked * self.l1.unsqueeze(0)).sum(dim=2))  # [B, layer_size²]
        y=apply_global_wta_inhibition(y,Config.OUTPUT_SPARSENESS)
        y=self.ln(y)
        if train_mode and self.training:
            with torch.no_grad():
                # Vectorized Oja update with masked Gabor inputs
                heb = (y.unsqueeze(2) * gabor_masked).mean(0)  # [layer_size², num_filters * H * W]
                dec = (y.pow(2).mean(0).unsqueeze(1) * self.l1)  # [layer_size², num_filters * H * W]
                self.l1.add_(Config.OJA_LR * (heb - dec))
                self.l1.div_(torch.norm(self.l1, dim=1, keepdim=True) + 1e-8)
        return y

# ---- Manhattan distance version ----
class SimplifiedVisNetLIMD(SimplifiedVisNetLI):
    def forward(self,x,train_mode=False):
        B=x.size(0)
        # Use parent's forward to get masked Gabor tensor, then override learning
        # Apply Gabor filters: [B, num_filters, H, W]
        f=[F.conv2d(x,g.unsqueeze(0),padding='same') for g in self.filters]
        gabor_tensor = torch.stack(f,dim=1)  # [B, num_filters, H, W]
        
        # Reshape to [B, num_filters * H * W] - keep all filters (don't average yet)
        num_filters = gabor_tensor.size(1)
        h, w = gabor_tensor.size(2), gabor_tensor.size(3)
        # Use -1 to auto-calculate flattened dimension (handles any spatial size)
        gabor_flat = gabor_tensor.view(B, -1)  # [B, num_filters * H * W]
        
        # Apply receptive field mask to full Gabor tensor (all filters × spatial)
        gabor_expanded = gabor_flat.unsqueeze(1)  # [B, 1, num_filters * H * W]
        gabor_masked = gabor_expanded * self.l1_mask.unsqueeze(0)  # [B, layer_size², num_filters * H * W]
        
        # Same as VisNet-LI: ReLU activation + Local inhibition
        y = torch.relu((gabor_masked * self.l1.unsqueeze(0)).sum(dim=2))  # [B, layer_size²]
        y=apply_local_inhibition_and_sparseness(y,Config.OUTPUT_SPARSENESS,
                    Config.RECEPTIVE_FIELD_RADIUS,self.layer_size,self.device)
        y=self.ln(y)
        if train_mode and self.training:
            with torch.no_grad():
                # Manhattan distance gradient learning (matching Run_Comparison_Experiment_VisNetMD11.py)
                # Use masked Gabor input: pp = masked Gabor tensor
                pp = gabor_masked  # [B, output_size, num_filters * H * W] - already masked
                w_expanded = self.l1.unsqueeze(0)  # [1, output_size, num_filters * H * W]
                # Apply mask to weights for consistency
                w_masked = w_expanded * self.l1_mask.unsqueeze(0)  # [B, output_size, num_filters * H * W]
                # Manhattan gradient: sign(w - x)
                grad = torch.sign(w_masked - pp)  # [B, output_size, num_filters * H * W]
                # Update: dw = lr * (grad - w) / batch_size
                dw = Config.MD_LR * (grad - w_masked).sum(dim=0) / B  # [output_size, num_filters * H * W]
                self.l1.add_(dw * self.l1_mask)  # Only update masked connections
                # Normalize weights after update
                norms = torch.norm(self.l1, dim=1, keepdim=True)
                self.l1.div_(norms + 1e-8)
        return y

# ---- RBF version ----
class SimplifiedVisNetLIRBF(BaseVisNet):
    def __init__(self,device,wid=2.0):
        super().__init__(device)
        # Use smaller width for squared distance in normalized space
        self.wid=wid / 100.0  # Scale down: 2.0 -> 0.02 for high-dim normalized space
        # Initialize weights from Gabor-like distribution, will be normalized by Oja
        with torch.no_grad():
            self.l1.data = torch.randn_like(self.l1) * 0.5  # Gaussian initialization
            # Initial normalization to unit sphere
            self.l1.div_(torch.norm(self.l1, dim=1, keepdim=True) + 1e-8)
    
    def _rbf(self,x_masked,w_masked): 
        # x_masked: [B, output_size, input_size], w_masked: [B, output_size, input_size]
        # Use squared distance with scaled width for stability
        dist_sq = ((x_masked-w_masked)**2).sum(2)  # [B, output_size]
        return torch.exp(-self.wid * dist_sq)
    
    def forward(self,x,train_mode=False):
        B=x.size(0)
        # Apply Gabor filters: [B, num_filters, H, W]
        f=[F.conv2d(x,g.unsqueeze(0),padding='same') for g in self.filters]
        gabor_tensor = torch.stack(f,dim=1)  # [B, num_filters, H, W]
        
        # Reshape to [B, num_filters * H * W]
        num_filters = gabor_tensor.size(1)
        h, w = gabor_tensor.size(2), gabor_tensor.size(3)
        gabor_flat = gabor_tensor.view(B, -1)  # [B, num_filters * H * W]
        
        # CRITICAL: Normalize Gabor responses to unit sphere (per sample)
        # This puts Gabor features in same scale as normalized weights
        gabor_flat = F.normalize(gabor_flat, dim=1, eps=1e-8)  # L2 normalize
        
        # Apply receptive field mask to normalized Gabor tensor
        gabor_expanded = gabor_flat.unsqueeze(1)  # [B, 1, num_filters * H * W]
        gabor_masked = gabor_expanded * self.l1_mask.unsqueeze(0)  # [B, layer_size², num_filters * H * W]
        w_masked = self.l1.unsqueeze(0) * self.l1_mask.unsqueeze(0)  # [B, layer_size², num_filters * H * W]
        
        y_raw = self._rbf(gabor_masked, w_masked)  # [B, layer_size²]
        y=apply_local_inhibition_and_sparseness(y_raw,Config.OUTPUT_SPARSENESS,
                    Config.RECEPTIVE_FIELD_RADIUS,self.layer_size,self.device)
        y=self.ln(y)
        
        if train_mode and self.training:
            with torch.no_grad():
                # Oja learning rule for RBF (on normalized Gabor features)
                # Hebbian term: y^T @ x (outer product averaged over batch)
                heb = (y_raw.unsqueeze(2) * gabor_masked).mean(0)  # [output_size, input_dim]
                
                # Anti-Hebbian decay term: y² * w
                dec = (y_raw.pow(2).mean(0).unsqueeze(1) * self.l1)  # [output_size, input_dim]
                
                # Oja update with receptive field mask
                self.l1.add_(Config.RBF_LR * (heb - dec) * self.l1_mask)
                
                # Normalize weights to unit sphere (standard Oja)
                self.l1.div_(torch.norm(self.l1, dim=1, keepdim=True) + 1e-8)
        return y

# ============================================================================
# TRAIN / TEST UTILITIES
# ============================================================================

def extract_features(model,img,device):
    if len(img.shape)==3: img=img.unsqueeze(0)
    if img.size(1)==3: img=0.2989*img[:,0:1]+0.587*img[:,1:2]+0.114*img[:,2:3]
    with torch.no_grad(): out=model(img.to(device)).cpu().numpy().flatten()
    return out

def run_single_experiment(train_size,method_name,dataset,random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
        torch.manual_seed(random_state)
    
    if method_name=="VisNet-LI": m=SimplifiedVisNetLI(Config.DEVICE)
    elif method_name=="VisNet-LI-DoG-RGB": m=SimplifiedVisNetLIDoGRGB(Config.DEVICE)
    elif method_name=="Simplified VisNet": m=SimplifiedVisNet(Config.DEVICE)
    elif method_name=="VisNet-MD": m=SimplifiedVisNetLIMD(Config.DEVICE)
    elif method_name=="VisNet-RBF": m=SimplifiedVisNetLIRBF(Config.DEVICE,Config.WID)
    else: raise ValueError("Unknown method")
    model=m.to(Config.DEVICE)

    labels=np.unique([l for _,l in dataset])
    c0=[d for d in dataset if d[1]==labels[0]]
    c1=[d for d in dataset if d[1]==labels[1]]
    np.random.shuffle(c0); np.random.shuffle(c1)
    train=c0[:train_size]+c1[:train_size]
    test =c0[train_size:train_size+Config.NUM_TEST_SAMPLES]+c1[train_size:train_size+Config.NUM_TEST_SAMPLES]

    loader=torch.utils.data.DataLoader(train,batch_size=Config.BATCH_SIZE,shuffle=True)

    model.train()
    for epoch in range(Config.NUM_EPOCHS):
        for imgs,_ in tqdm(loader,desc=f"Epoch {epoch+1}",leave=False):
            imgs=imgs.to(Config.DEVICE)
            model(imgs,train_mode=True)
        # optional small norm decay
        with torch.no_grad(): model.l1.mul_(0.99)

    model.eval()
    Xtr=[extract_features(model,img,Config.DEVICE) for img,_ in train]
    ytr=[lbl for _,lbl in train]
    Xte=[extract_features(model,img,Config.DEVICE) for img,_ in test]
    yte=[lbl for _,lbl in test]

    clf=LinearSVC(max_iter=10000,dual=False)
    clf.fit(Xtr,ytr)
    acc=accuracy_score(yte,clf.predict(Xte))
    return acc

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*60)
    print("VisNet Optimized Experiment")
    print("Device:",Config.DEVICE)
    print("="*60)
    
    # Resolve dataset path
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        os.path.join(SCRIPT_DIR, "Claltech101", "101_ObjectCategories"),
        os.path.join(SCRIPT_DIR, "Caltech101", "101_ObjectCategories"),
        os.path.join(SCRIPT_DIR, "..", "..", "Claltech101", "101_ObjectCategories"),
        os.path.join(SCRIPT_DIR, "..", "..", "Caltech101", "101_ObjectCategories"),
        Config.DATASET_PATH if Config.DATASET_PATH and os.path.exists(Config.DATASET_PATH) else None
    ]
    dataset_path = None
    for path in possible_paths:
        if path and os.path.exists(path):
            dataset_path = path
            break
    
    if not dataset_path:
        print("Dataset not found. Tried:")
        for path in possible_paths:
            if path:
                print(f"  - {path}")
        print("Please set Config.DATASET_PATH to a valid ImageFolder.")
        return
    
    # Use different transforms for RGB vs grayscale models
    transform_gray=transforms.Compose([
        transforms.Resize((32,32)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,),(0.5,))
    ])
    
    transform_rgb=transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    
    # For now, use grayscale as default (we'll handle RGB separately for VisNet-LI-DoG-RGB)
    transform = transform_gray
    dataset_gray=datasets.ImageFolder(dataset_path,transform=transform_gray)
    dataset_rgb=datasets.ImageFolder(dataset_path,transform=transform_rgb)
    print("Loaded dataset with",len(dataset_gray),"samples")
    methods=["Simplified VisNet","VisNet-LI","VisNet-LI-DoG-RGB","VisNet-MD","VisNet-RBF"]
    results={}
    
    for m in methods:
        print(f"\n{'='*60}")
        print(f"METHOD: {m}")
        print(f"{'='*60}")
        
        # Use RGB dataset for VisNet-LI-DoG-RGB, grayscale for others
        dataset = dataset_rgb if m == "VisNet-LI-DoG-RGB" else dataset_gray
        
        method_results={}
        for s in Config.TRAIN_SIZES:
            print(f"\nTraining size: {s} samples per class")
            accuracies = []
            for trial in range(Config.NUM_TRIALS):
                acc = run_single_experiment(s, m, dataset, random_state=trial*42 + 10)
                accuracies.append(acc)
                print(f"  Trial {trial+1}/{Config.NUM_TRIALS}: {acc*100:.2f}%")
            
            mean_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)
            method_results[s] = {'mean': mean_acc, 'std': std_acc, 'all': accuracies}
            print(f"  → Mean: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
        results[m] = method_results

    # Baseline values from Rolls 2015 Figure 4
    paper_train_sizes = [5, 15, 30]
    hmax_acc = [0.68, 0.60, 0.66]  # HMAX C2 performance
    hmax_std = [0.15, 0.08, 0.08]  # HMAX standard deviations
    original_visnet_acc = [0.49, 0.61, 0.59]  # Original VisNet performance
    original_visnet_std = [0.04, 0.04, 0.04]  # Original VisNet standard deviations
    
    plt.figure(figsize=(12,7))
    
    # Plot HMAX baseline (from Rolls 2015)
    plt.errorbar(paper_train_sizes, np.array(hmax_acc)*100, yerr=np.array(hmax_std)*100,
                fmt='-o', capsize=5, linewidth=2.5, markersize=8, color='red', alpha=0.9,
                label='HMAX C2 (Rolls 2015)', markeredgewidth=1, markeredgecolor='white')
    
    # Plot Original VisNet baseline (from Rolls 2015)
    plt.errorbar(paper_train_sizes, np.array(original_visnet_acc)*100, yerr=np.array(original_visnet_std)*100,
                fmt='--s', capsize=5, linewidth=2.5, markersize=8, color='blue', alpha=0.9,
                label='Original VisNet (Rolls 2015)', markeredgewidth=1, markeredgecolor='white')
    
    # Plot our implementations with error bars (distinct colors)
    colors = ['darkgreen', 'darkorange', 'purple', 'cyan', 'brown', 'deeppink']
    for idx, (m, res) in enumerate(results.items()):
        xs = sorted(res.keys())
        ys = [res[s]['mean']*100 for s in xs]
        stds = [res[s]['std']*100 for s in xs]
        plt.errorbar(xs, ys, yerr=stds, fmt='-o', capsize=5, linewidth=2, markersize=8,
                    label=m, color=colors[idx % len(colors)], alpha=0.9, markeredgewidth=1, markeredgecolor='white')
    
    plt.xlabel("Training samples per class"); plt.ylabel("Accuracy (%)")
    plt.title("VisNet variant comparison (with baseline methods from Rolls 2015)")
    plt.legend(); plt.grid(True, alpha=0.3); plt.show()

if __name__=="__main__":
    main()
