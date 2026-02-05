import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from scipy.ndimage import convolve
import torch
import torch.nn as nn
import torch.optim as optim
#from numba import jit
import torch.nn.functional as F
import cv2
from multiprocessing import freeze_support
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split


#@jit(nopython=True)
#def gabor_kernel(frequency, theta, sigma=10, lambd=10, gamma=0.5, psi=0):
#def gabor_kernel(frequency, theta, sigma=5, lambd=10, gamma=1, psi=0):
#def gabor_kernel(frequency, theta, sigma=4.0, lambd=10, gamma=0.5, psi=0):
def gabor_kernel(frequency, theta, sigma=5.0, lambd=10, gamma=1, psi=0):
#def gabor_kernel(frequency, theta, sigma=5.0, lambd=1, gamma=1, psi=0):
    
    
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
    """
    Normalize the Gabor kernel to ensure the sum of absolute values equals 1
    """
    return kernel / np.sum(np.abs(kernel))
    
    
    
    
#@jit(nopython=True)
def generate_gabor_filters(frequencies, orientations, phases):
    filters = []
    for frequency in frequencies:
        for theta in orientations:
            for psi in phases:
                kernel = gabor_kernel(frequency=frequency, theta=theta, psi=psi)
                filters.append(normalize_kernel(kernel))
    return filters



#@jit(nopython=True)
def apply_gabor_filters(image, filters):
    filtered_images = np.array([convolve(image, kernel, mode='reflect') for kernel in filters])
    return np.mean(filtered_images, axis=0)  # Averaging filter responses



# Function to apply Gabor filters to an image
def apply_gabor_filters2(image, filters):
    # Ensure image is a 2D numpy array
    if image.ndim == 3:  # If the image is 3D (color), convert to 2D (grayscale)
        image = image.mean(axis=2)
    filtered_images = np.array([convolve(image, kernel, mode='reflect') for kernel in filters])
    # Aggregate filter responses, e.g., by taking the maximum response at each pixel location
    aggregated_response = filtered_images.max(axis=0)
    return aggregated_response
    


# Function to apply Gabor filters to an image
def apply_gabor_filters3(image, filters):
    # Ensure image is a 2D numpy array
    if image.ndim == 3:  # If the image is 3D (color), convert to 2D (grayscale)
        image = image.mean(axis=2)
    filtered_images = np.array([convolve(image, kernel, mode='reflect') for kernel in filters])
    # Aggregate filter responses, e.g., by taking the maximum response at each pixel location
    aggregated_response = filtered_images.max(axis=0)
    return aggregated_response




def apply_gabor_filters4_opencv(image, filters):
    if image.dtype != np.float32:
        image = image.astype(np.float32)
    
    if image.ndim == 3 and image.shape[2] in [3, 4]:  # Check for color or color+alpha
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    filtered_images = []
    for kernel in filters:
        if kernel.dtype != np.float32:
            kernel = kernel.astype(np.float32)
        kernel_flipped = kernel[::-1, ::-1]
        filtered_img = cv2.filter2D(image, -1, kernel_flipped, borderType=cv2.BORDER_REFLECT)
        filtered_images.append(filtered_img)
    
    return np.array(filtered_images)




def preprocess_images(images, gabor_filters):
    processed_images = []
    for img in images:
        # Check and convert if not a numpy array
        if not isinstance(img, np.ndarray):
            if hasattr(img, 'convert'):  # Check if it's a PIL Image
                img = np.array(img.convert('L'))  # Convert PIL Image to grayscale numpy array
            else:
                img = np.array(img)  # Try converting to numpy array directly
        
        img = np.squeeze(img)
        if img.ndim not in [2, 3]:  # Valid dimensions check
            raise ValueError("Image must be 2D or 3D")

        processed_img = apply_gabor_filters4_opencv(img, gabor_filters)
        #print(processed_img)
        processed_images.append(processed_img)
    
    return np.array(processed_images)
    
    
    

    
#@jit(nopython=True)    
def sigmoid(x):
    return 1 / (1 + np.exp(-x))





def load_cifar10_data(batch_size):
    transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
    train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader





#@jit(nopython=True)
def normVecLen(vec):
    """Normalize the vectors to unit length."""
    norm = np.linalg.norm(vec, axis=0, keepdims=True)
    return vec / norm




def min_max_scale(tensor, axis=1):
 
 
  min_vals, _ = torch.min(tensor, dim=axis, keepdim=True)  # Find min along specified axis, keep dimensions
  max_vals, _ = torch.max(tensor, dim=axis, keepdim=True)  # Find max along specified axis, keep dimensions
  scaled_tensor = (tensor - min_vals) / (max_vals - min_vals)
  
  return scaled_tensor




def create_mask_batch(x, y, h, w,radius):

    all_mask = torch.zeros(x, y, h, w)
    
    for i in range(x):
            for j in range(y):
                #mask = create_circular_mask(h, w, center=(x, y), radius=radius)
                #mask = create_binary_gaussian_mask(h, w, center=(i, j), radius=radius)
                all_mask[i, j, : ,:]=create_binary_gaussian_mask(h, w, center=(i, j), radius=radius)
                

    return all_mask
    
    
    

def create_mask_batch2(x, y, h, w,radius):

    all_mask = torch.zeros(x, y, h, w)
    
    for i in range(x):
            for j in range(y):
                #mask = create_circular_mask(h, w, center=(x, y), radius=radius)
                #mask = create_binary_gaussian_mask(h, w, center=(i, j), radius=radius)
                all_mask[i, j, : ,:]=create_circular_mask(h, w, center=(i, j), radius=radius)
                

    return all_mask
    


  
def create_mask_batch3(x, y, h, w,radius):

    all_mask = torch.zeros(x, y, h, w)
    
    for i in range(x):
            for j in range(y):
                #mask = create_circular_mask(h, w, center=(x, y), radius=radius)
                #mask = create_binary_gaussian_mask(h, w, center=(i, j), radius=radius)
                all_mask[i, j, : ,:]=create_binary_gaussian_mask2(h, w, center=(i, j), radius=radius)
                

    return all_mask





def normalize(tensor):
  """Normalizes a PyTorch tensor to the range [0, 1].

  Args:
    tensor: The input PyTorch tensor.

  Returns:
    The normalized PyTorch tensor.
  """

  min_val = torch.min(tensor)
  max_val = torch.max(tensor)
  return (tensor - min_val) / (max_val - min_val)







def compute_activations_2d_circular(input_data, SynMat):
  
  #print(input_data.shape)
  input_data = input_data.view(batch_size, layer_size[0]*layer_size[1])
  input_data = input_data .unsqueeze(1)
  input_data = input_data.repeat(1,layer_size[0]*layer_size[1],1)
  
  #print(input_data.shape)
  #print(all_mask.shape)
  
  
  pp = input_data*all_mask
  
  ww = SynMat
  
  activations = torch.mul(ww,pp).sum(dim=(2))
  
  activations = normalize(activations)
  #return torch.sigmoid(activations)
  #return torch.nn.functional.relu(activations)
  return activations



def compute_activations_2d_circular_test(input_data, SynMat):
  
  #print(input_data.shape)
  input_data = input_data.view(batch_size, layer_size[0]*layer_size[1])
  input_data = input_data .unsqueeze(1)
  input_data = input_data.repeat(1,layer_size[0]*layer_size[1],1)
  
  #print(input_data.shape)
  #print(all_mask.shape)
  
  
  pp = input_data*all_mask
  
  ww = SynMat
  
  activations = torch.mul(ww,pp).sum(dim=(2))
  
  #activations = normalize(activations)
  #return torch.sigmoid(activations)
  return torch.nn.functional.relu(activations)
  #return activations



def compute_activations_2d_circular_L1(input_data, SynMat):
  
  #print(input_data.shape)
  input_data = torch.tensor(input_data)
  input_data = input_data.view(batch_size, nf*input_size[0]*input_size[1])
  #input_data = input_data.reshape(batch_size, 32*input_size[0]*input_size[1])
  input_data = input_data .unsqueeze(1)
  input_data = input_data.repeat(1,layer_size[0]*layer_size[1],1)
  
  input_data=input_data.to(device)
  #print(input_data.shape)
  #print(all_mask_L1.shape)
  
  
  pp = input_data*all_mask_L1
  
  ww = SynMat
  
  activations = torch.mul(ww,pp).sum(dim=(2))
  
  activations = normalize(activations)
  
  #return torch.sigmoid(activations)
  #return torch.nn.functional.relu(activations)
  return activations


  
def compute_activations_2d_circular_L1_test(input_data, SynMat):
  
  #print(input_data.shape)
  input_data = torch.tensor(input_data)
  input_data = input_data.view(batch_size, nf*input_size[0]*input_size[1])
  #input_data = input_data.reshape(batch_size, 32*input_size[0]*input_size[1])
  input_data = input_data .unsqueeze(1)
  input_data = input_data.repeat(1,layer_size[0]*layer_size[1],1)
  
  input_data=input_data.to(device)
  #print(input_data.shape)
  #print(all_mask_L1.shape)
  
  
  pp = input_data*all_mask_L1
  
  ww = SynMat
  
  activations = torch.mul(ww,pp).sum(dim=(2))
  
  #activations = normalize(activations)
  
  #return torch.sigmoid(activations)
  return torch.nn.functional.relu(activations)
  #return activations



# def apply_local_inhibition_and_sparseness(activations, inhibition_strength=1.0):
    
    
    # batch,fs=activations.shape
    
    # inhibited_activations = torch.zeros_like(activations)
    
    # for b in range(batch):
    
        
        # threshold = torch.kthvalue(activations[:, b], int((1 - inhibition_strength) * activations[:, b].numel())).values
                
        # for i in range(fs):
           
                
            # if activations[b,i]>=threshold:
                # inhibited_activations[b,i] = activations[b,i]
            # else:
                # inhibited_activations[b,i] = -1*activations[b,i]

    # return inhibited_activations
    
    
    
    
    
def apply_local_inhibition_and_sparseness(activations, inhibition_strength=1.0):
    
    
    #batch,fs=activations.shape
    
    inhibited_activations = torch.zeros_like(activations)
    
    
    # Assuming inhibited_activations is empty
    inhibited_activations = torch.where(
        # Calculate threshold per column (assuming k-th largest per column)
        torch.kthvalue(activations, int((1 - inhibition_strength) * activations.shape[1]))[0].unsqueeze(1) 
        <= activations,
        activations,
        -0.0*activations)



    #random_tensor = torch.rand_like(inhibited_activations)
    
    
    #mask = random_tensor<=inhibited_activations    


    #return inhibited_activations*mask
    return inhibited_activations
    




def create_circular_mask(h, w, center=None, radius=None):
    
    
    if center is None:  # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None:  # use the smallest distance between the center and image borders
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)

    mask = dist_from_center <= radius
    return torch.from_numpy(mask).float()





def create_binary_gaussian_mask(h, w, center=None, radius=None, global_sigma=1):
    
    
    if center is None:
        center = (w / 2, h / 2)
    if radius is None:
        radius = min(center[0], center[1], w - center[0], h - center[1])

    Y, X = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    dist_from_center = torch.sqrt((X - center[0])**2 + (Y - center[1])**2)

    # Adjust sigma based on distance from center: lower sigma closer to center
    sigma_map = global_sigma * (dist_from_center / radius + 1)  # Increase sigma with distance
    sigma_map = torch.clamp(sigma_map, min=0.1, max=global_sigma)  # Ensure sigma is positive and does not exceed global_sigma

    # Sample from Gaussian distributions with varying sigma
    random_values = torch.normal(mean=0, std=sigma_map)

    # Apply a threshold to create a binary mask: here using 0 as a threshold for simplicity
    binary_mask = random_values > 0  # Threshold at mean of normal distribution
    return binary_mask.float()




def create_binary_gaussian_mask2(h, w, center=None, radius=None, global_sigma=1):
    
    
    if center is None:
        center = (w / 2, h / 2)
    if radius is None:
        radius = min(center[0], center[1], w - center[0], h - center[1])

    Y, X = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    dist_from_center = torch.sqrt((X - center[0])**2 + (Y - center[1])**2)

    # Adjust sigma based on distance from center: lower sigma closer to center
    sigma_map = global_sigma * (dist_from_center / radius + 1)  # Increase sigma with distance
    sigma_map = torch.clamp(sigma_map, min=0.1, max=global_sigma)  # Ensure sigma is positive and does not exceed global_sigma

    # Sample from Gaussian distributions with varying sigma
    random_values = torch.normal(mean=0, std=sigma_map)

    # Apply a threshold to create a binary mask: here using 0 as a threshold for simplicity
    #binary_mask = random_values > 0  # Threshold at mean of normal distribution
    binary_mask = random_values  # Threshold at mean of normal distribution
    
    return binary_mask.float()




def compute_covariance(tensor1, tensor2):
    """
    Compute the covariance between two 1D tensors.

    :param tensor1: A 1D tensor.
    :param tensor2: A 1D tensor.
    :return: Covariance between tensor1 and tensor2.
    """
    if tensor1.dim() != 1 or tensor2.dim() != 1:
        raise ValueError("Both tensors must be 1-dimensional")

    if tensor1.size(0) != tensor2.size(0):
        raise ValueError("Both tensors must have the same length")

    n = tensor1.size(0)
    mean1 = torch.mean(tensor1)
    mean2 = torch.mean(tensor2)

    cov = torch.sum((tensor1 - mean1) * (tensor2 - mean2)) / (n - 1)
    return cov
    

    
# Function to compute the gradient of the Manhattan distance
def gradient_distance(x, w):
    return torch.sign(w - x)



def update_weights_2d_circular2(SynMat, Rate_old, input_data, Rate, Learnrate, alpha):
    
    input_data = input_data.view(batch_size, layer_size[0]*layer_size[1])
    input_data = input_data .unsqueeze(1)
    input_data = input_data.repeat(1,layer_size[0]*layer_size[1],1)
    
    #Rate_4D = Rate.view(batch_size, layer_size[0], layer_size[1])  # Hypothetical reshaping, replace with your logic
    Rate_4D = Rate
    
    if iter:
        Rate_4D= (1-alpha)*Rate_4D+ (alpha)*Rate_old
    else:
        Rate_4D= (1-alpha)*Rate_4D
    
    aa=Rate_4D
    pp=input_data*all_mask
    
    
    
    
    
    aa = aa .unsqueeze(2)
    aa = aa.repeat(1,1,layer_size[0]*layer_size[1])
    aa = aa*all_mask
    
    
    #adot = aa*(aa>0)
    
    
    
    
    
    #pp = pp.view(-1)
    #aa = aa.view(-1)
    #pa = compute_covariance(pp, aa)    
    
    
    #print(pp.shape)
    #print(aa.shape)
    #print(SynMat.shape)
    
    #pa=torch.mul(pp*pp,adot).sum(dim=0)
    
    # #print(pp.shape)
    # #print(aa.shape)
    # #print(SynMat.shape)
    # #SynMat_updated=elementwise_covariance_rule2(SynMat, pp, aa, Learnrate)
        
    # pa=torch.mul(pp,aa).sum(dim=0)
    # #print(pa.shape)
    # dw=Learnrate*(pa-SynMat)/batch_size
    
    
    #pp = pp.view(-1)
    #aa = aa.view(-1)
    #pa = compute_covariance(pp, aa)    
    
    
    #print(pp.shape)
    #print(aa.shape)
    #print(SynMat.shape)
    #pa=torch.mul(pp,aa).sum(dim=0)
    #print(pa.shape)



    grad = gradient_distance(pp, SynMat)
    
    
    dw=Learnrate*(grad-SynMat)/batch_size
    
    
    #dw=Learnrate*aa*(grad-SynMat)/batch_size
    #dw=Learnrate*(dw-SynMat)/batch_size
    #dw=Learnrate*(grad)/batch_size
    
    
    SynMat_updated = SynMat+ dw
    
    # Normalize the new weights
    
    norm = torch.linalg.vector_norm(SynMat_updated, dim=1, keepdim=True)
    SynMat_updated = SynMat_updated / norm
    
    
    #return (SynMat_updated+SynMat)/2
    return SynMat_updated




def update_weights_2d_circular2_L1(SynMat, Rate_old, input_data, Rate, Learnrate, alpha):
    
    input_data = input_data.view(batch_size, nf*input_size[0]*input_size[1])
    input_data = input_data .unsqueeze(1)
    input_data = input_data.repeat(1,layer_size[0]*layer_size[1],1)
  
 
    #Rate_4D = Rate.view(batch_size, layer_size[0], layer_size[1])  # Hypothetical reshaping, replace with your logic
    Rate_4D = Rate
    
    if iter:
        Rate_4D= (1-alpha)*Rate_4D+ (alpha)*Rate_old
    else:
        Rate_4D= (1-alpha)*Rate_4D
    
    aa=Rate_4D
    
    input_data=input_data.to(device)
    
    pp=input_data*all_mask_L1
    
    
    
    # print(aa.shape)
    # print(all_mask_L1.shape)
    
    aa = aa .unsqueeze(2)
    aa = aa.repeat(1,1,nf*input_size[0]*input_size[1])
    aa = aa*all_mask_L1
    
    #adot =aa*(aa>0)
    
    
    
    
    
    #pp = pp.view(-1)
    #aa = aa.view(-1)
    #pa = compute_covariance(pp, aa)    
    
    
    #print(pp.shape)
    #print(aa.shape)
    #print(SynMat.shape)
    grad = gradient_distance(pp, SynMat)
   
    #dw=Learnrate*aa*(grad-SynMat)/batch_size
    dw=Learnrate*(grad-SynMat)/batch_size
    
    #dw=Learnrate*(grad)/batch_size
    #dw=Learnrate*(dw-SynMat)/batch_size
    
    SynMat_updated = SynMat+ dw
    
    
    # Normalize the new weights
    norm = torch.linalg.vector_norm(SynMat_updated, dim=1, keepdim=True)
    SynMat_updated = SynMat_updated / norm
    
    #return (SynMat_updated+SynMat)/2
    return SynMat_updated

    


def competitive_layer_2d(input_data, Rate_old, SynMat, Learnrate, OutputSparseness, alpha, layer):
    
    
    input_data = torch.tensor(input_data, dtype=torch.float32)
    SynMat = torch.tensor(SynMat, dtype=torch.float32)

    
    if layer==1:
        # Compute activations for the 2D layer
        Actvn = compute_activations_2d_circular_L1(input_data, SynMat)
    else:
        Actvn = compute_activations_2d_circular(input_data, SynMat)


    if layer==1:
        Actvn = apply_local_inhibition_and_sparseness(Actvn, OutputSparseness)
    else:
        Actvn = apply_local_inhibition_and_sparseness(Actvn, OutputSparseness)
        
        

    if layer==1:
        # Update weights (Placeholder)
        SynMat_updated = update_weights_2d_circular2_L1(SynMat, Rate_old, input_data, Actvn, Learnrate, alpha)
    else:
        SynMat_updated = update_weights_2d_circular2(SynMat, Rate_old, input_data, Actvn, Learnrate, alpha)
        
        
    #return F.leaky_relu(Actvn), SynMat_updated
    #return F.relu(Actvn), SynMat_updated
    return Actvn, SynMat_updated



    
def compute_classification_weights(activations, labels):
    #labels_one_hot = np.eye(10)[labels]  # CIFAR-10 has 10 classes
    
    weights = torch.matmul(torch.linalg.pinv(activations), labels.float())
    return weights




def predict_and_evaluate(loader, gabor_filters, SynMat_layers, weights):
    all_activations = []
    all_labels = []
    
    for images, labels in loader:
        images = preprocess_images(images, gabor_filters)  # Apply Gabor filters
        
        # Sequentially process through competitive layers
        activations = images
        for SynMat in SynMat_layers:
            activations = sigmoid(np.dot(activations, SynMat))  # Assuming sigmoid activation
        
        all_activations.append(activations)
        all_labels.extend(labels.numpy())
    
    all_activations = np.vstack(all_activations)
    predictions = np.dot(all_activations, weights)
    predicted_labels = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(all_labels, predicted_labels)
    
    return accuracy




 
 
#@jit(nopython=True)
def collect_activations_and_labels(loader, gabor_filters, SynMat_layers):
    all_activations = []
    all_labels = []
    
    for images, labels in loader:
        # Convert images to numpy if they're in tensors
        if isinstance(images, torch.Tensor):
            images = images.numpy()
        
        # Preprocess and apply Gabor filters to each image in the batch
        processed_images = preprocess_images(images, gabor_filters)
        
        # Sequentially process the images through the competitive layers
        activations = processed_images
        for SynMat in SynMat_layers:
            # Assuming competitive_layer function processes each image and returns activations
            # Update to match your actual function's signature and return value
            activations, _ = competitive_layer(activations, SynMat, Learnrate, OutputSparseness, alpha=0.0)  # Example
        
        # Collect the final layer activations and the labels
        all_activations.append(activations)
        all_labels.extend(labels.numpy())  # Convert labels to numpy if they're in tensors
    
    # Stack all collected activations into a single numpy array
    all_activations = np.vstack(all_activations)
    all_labels = np.array(all_labels)
    
    return all_activations, all_labels





#@jit(nopython=True)
def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs.data, 0)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    return correct / total





def initialize_weights(shape, method='xavier'):
    if method == 'xavier':
        bound = np.sqrt(6 / (shape[0] + shape[1]))
        return np.random.uniform(-bound, bound, shape)
    elif method == 'he':
        std = np.sqrt(2 / shape[0])
        return np.random.normal(0, std, shape)
    elif method == 'orthogonal':
        # Only works if the weight matrix is square
        from scipy.linalg import orth
        X = np.random.random(shape)
        return orth(X)
    else:
        raise ValueError("Unsupported initialization method")





def process_images_through_competitive_layers_2d_circular(images, gabor_filters, OutputSparseness, SynMat1, SynMat2, SynMat3, SynMat4):


    # Ensure input_data and SynMat are torch tensors
    images = torch.tensor(images, dtype=torch.float32)
    #SynMat = torch.tensor(SynMat, dtype=torch.float32)

    gabor_processed = preprocess_images(images, gabor_filters)
    
    # Compute activations for the 2D layer
    Actvn = compute_activations_2d_circular_L1(gabor_processed, SynMat1)
    
    
    Actvn = apply_local_inhibition_and_sparseness(Actvn, OutputSparseness)
    
    
    Actvn = compute_activations_2d_circular(Actvn, SynMat2)
    
    
    Actvn = apply_local_inhibition_and_sparseness(Actvn, OutputSparseness)
    
    
    Actvn = compute_activations_2d_circular(Actvn, SynMat3)
    
    
    Actvn = apply_local_inhibition_and_sparseness(Actvn, OutputSparseness)
    
    
    Actvn = compute_activations_2d_circular(Actvn, SynMat4)
    
    
    Actvn = apply_local_inhibition_and_sparseness(Actvn, OutputSparseness)
    
    return Actvn




class ClassificationLayer(nn.Module):
    def __init__(self, input_size, output_size=10):
        super(ClassificationLayer, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        return self.linear(x)


# ----------------------------------------------MAIN---------------------------------------------------
# CIFAR-10 preprocessing
# =============================================================================
# transform = transforms.Compose([
#     transforms.Resize((32, 32)),
#     transforms.Grayscale(),  # Convert images to grayscale
#     transforms.ToTensor(),
#     #transforms.RandomHorizontalFlip(),           # Randomly flip the image horizontally
#     #transforms.RandomRotation(10),               # Randomly rotate the image by up to 10 degrees
#     #transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Randomly translate the image by up to 10% of its size
#     #transforms.Normalize((0.5,), (0.5,))  # Normalize images
#     transforms.Normalize((0.1307,), (0.3081,))
# ])
# =============================================================================


# CIFAR-10 preprocessing
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(),  # Convert images to grayscale
    transforms.ToTensor(),
    #transforms.RandomHorizontalFlip(),           # Randomly flip the image horizontally
    #transforms.RandomRotation(180),               # Randomly rotate the image by up to 10 degrees
    #transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),  # Randomly translate the image by up to 10% of its size
    transforms.Normalize((0.5,), (0.5,))  # Normalize images
    #transforms.Normalize((0.1307,), (0.3081,))
])

# =============================================================================
# 
# =============================================================================


batch_size = 1
n_epochs = 1
#Learnrate = 0.001
InputSparseness = 0.01
alpha = 0.8
alpha1 = 0.0


# Define transformations
# # transform = transforms.Compose([
    # transforms.RandomResizedCrop(224),
    # transforms.RandomHorizontalFlip(),
    # # transforms.ToTensor(),
    # # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# # ])


# # Load the dataset
# #data_dir = r'C:\Users\mfatan\Desktop\visnet\sierpinski_triangles_dataset9\sierpinski_triangles_dataset9'
# data_dir = r'C:\Users\mfatan\Desktop\visnet\archive\train2'
# #data_dir = r'C:\Users\mfatan\Desktop\visnet\SYMMETRY_PROJECT\SYMMETRY_PROJECT\SYMMETRY_Project'
# train_dataset = datasets.ImageFolder(root=data_dir, transform=transform)



# Load the dataset
#data_dir = r'C:\Users\mfatan\Desktop\visnet\sierpinski_triangles_dataset'
#data_dir = r'C:\Users\mfatan\Desktop\visnet\sierpinski_triangles_dataset8'
# data_dir = r'C:\Users\mfatan\Desktop\visnet\multi_asymmetric_square_dataset'
# #data_dir = r'C:\Users\mfatan\Desktop\visnet\SYMMETRY_PROJECT\SYMMETRY_PROJECT\SYMMETRY_Project'
# full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# # Split the dataset into training and validation sets
# train_size = int(0.8 * len(full_dataset))
# test_size = len(full_dataset) - train_size
# train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

# # Create dataloaders
# #batch_size = 32

# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)




# # Transformation pipeline for preprocessing
# transform = transforms.Compose([
#     transforms.ToTensor(),  # Convert images to PyTorch tensors
#     transforms.Normalize((0.1307,), (0.3081,))  # Normalize the dataset using the mean and std dev of MNIST
# ])


# Loading the MNIST training dataset
trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                      download=True, transform=transform)

                                      
# Loading the MNIST test dataset
testset = torchvision.datasets.MNIST(root='./data', train=False,
                                     download=True, transform=transform)


# DataLoader setup for both training and testing
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=0)
                                          
                 
                 
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=0)





# # Transformation pipeline for preprocessing CIFAR-10
# transform = transforms.Compose([
#     transforms.ToTensor(),  # Convert images to PyTorch tensors
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # Normalize using CIFAR-10 dataset's mean and std dev
#     #transforms.Grayscale(),  # Convert images to grayscale
#     transforms.Resize((32, 32)),
# ])

# # Loading the CIFAR-10 training dataset
# trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                         download=True, transform=transform)

# # Loading the CIFAR-10 test dataset
# testset = torchvision.datasets.CIFAR10(root='./data', train=False,
#                                        download=True, transform=transform)

# # DataLoader setup for both training and testing
# train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
#                                            shuffle=True, num_workers=0)

# test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
#                                           shuffle=False, num_workers=0)






# Network parameters
N = 6000  # Number of neurons per layer
Learnrate = 0.0000001
Learnrate_L1 = 0.000000005
OutputSparseness = 0.999


Learnrate = 0.1
Learnrate_L1 = 0.05


Learnrate = 0.0005
Learnrate_L1 = 0.0025



num_clases = 10
#batch_size=2561
device = torch.device("cuda") 

# Define a softmax layer
softmax_layer = nn.Softmax(dim=0)


# Example parameters
layer_size = (32, 32)  # 10x10 neurons in the layer
input_size = (32, 32)  # For MNIST-sized input images

receptive_field_radius1 = 15  # Radius of the circular receptive field
receptive_field_radius2 = 12  # Radius of the circular receptive field

receptive_field_radius1 = 7  # Radius of the circular receptive field
receptive_field_radius2 = 7  # Radius of the circular receptive field


receptive_field_radius1 = 7  # Radius of the circular receptive field
receptive_field_radius2 = 7  # Radius of the circular receptive field


#frequencies = [0.2, 0.4, 0.6, 0.8]
frequencies = [0.0625, 0.125, 0.25, 0.5]
orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4]
#phases = [0, np.pi]
#phases = [-1*np.pi, np.pi]
phases = [-1.0, 1.0]



#frequencies = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]  # More frequencies for finer detail
#orientations = np.linspace(0, np.pi, 10)  # 8 orientations for better directional coverage
#phases = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]  # More phase shifts for greater variety


nf=len(frequencies)*len(orientations)*len(phases)

gabor_filters = generate_gabor_filters(frequencies, orientations, phases)


fcn_neurons = layer_size[0] * layer_size[1]  # Assuming grayscale images after Gabor filtering


SynMat1 = torch.from_numpy(initialize_weights((layer_size[0]*layer_size[1], nf*input_size[0]*input_size[1]), 'xavier')).to(device)
SynMat2 = torch.from_numpy(initialize_weights((layer_size[0]*layer_size[1], layer_size[0]*layer_size[1]), 'xavier')).to(device)
SynMat3 = torch.from_numpy(initialize_weights((layer_size[0]*layer_size[1], layer_size[0]*layer_size[1]), 'xavier')).to(device)
SynMat4 = torch.from_numpy(initialize_weights((layer_size[0]*layer_size[1], layer_size[0]*layer_size[1]), 'xavier')).to(device)

# Assuming SynMat1, SynMat2, SynMat3, SynMat4 are the synaptic matrices after training
#SynMat_layers = [SynMat1, SynMat2, SynMat3, SynMat4]


# Assuming 100 features from the last competitive layer
#input_size = N  # This should match the output size of your last competitive layer
classification_layer = ClassificationLayer(fcn_neurons, output_size=10).to(device)

#optimizer = optim.SGD(classification_layer.parameters(), lr=0.01, momentum=0.9)
#optimizer = optim.Adam(classification_layer.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()  # CIFAR-10 classification uses CrossEntropy


Rate1= torch.from_numpy(np.random.rand(layer_size[0], layer_size[1])).to(device)
Rate2= torch.from_numpy(np.random.rand(layer_size[0], layer_size[1])).to(device)
Rate3= torch.from_numpy(np.random.rand(layer_size[0], layer_size[1])).to(device)
Rate4= torch.from_numpy(np.random.rand(layer_size[0], layer_size[1])).to(device)


# Example parameters
# layer_size = (80, 80)  # 10x10 neurons in the layer
# input_size = (42, 42)  # For MNIST-sized input images
x=layer_size[0]
y=layer_size[1]
w=input_size[0]
h=input_size[1]


# all_mask_L1 = create_mask_batch3(x, y, 32*w, h, receptive_field_radius1)
# all_mask = create_mask_batch3(x, y, x, y, receptive_field_radius2)



# all_mask_L1 = all_mask_L1.view(x*y, 32*w*h)
# all_mask = all_mask.view(x*y, x*y)


# all_mask_L1 = torch.unsqueeze(all_mask_L1, 0)
# all_mask_L1 = all_mask_L1.repeat(batch_size, 1, 1).to(device)


# all_mask = torch.unsqueeze(all_mask, 0)
# all_mask = all_mask.repeat(batch_size, 1, 1).to(device)


#-------------------------------------------------------

all_mask_L1 = create_mask_batch2(x, y, nf*w, h, receptive_field_radius1)
all_mask = create_mask_batch2(x, y, x, y, receptive_field_radius2)



all_mask_L1 = all_mask_L1.view(x*y, nf*w*h)
all_mask = all_mask.view(x*y, x*y)


all_mask_L1 = torch.unsqueeze(all_mask_L1, 0)
all_mask_L1 = all_mask_L1.repeat(batch_size, 1, 1).to(device)


all_mask = torch.unsqueeze(all_mask, 0)
all_mask = all_mask.repeat(batch_size, 1, 1).to(device)




running_loss = 0.0
running_accuracy = 0.0
classification_layer.train()  # Set the network to training mode
all_outputs=[]
all_labels=[]

lim=8000


for epoch in range(n_epochs):
    
    
    for iter, data in enumerate(train_loader, 0):
        images, labels = data
        
        labels = labels.to(device)
        # Assuming your processing function returns a NumPy array, convert it to tensor
        
        #OutputSparseness = min(np.random.rand(),0.01)
        
        if iter>lim:
            break
        
        
        
        gabor_processed = preprocess_images(images, gabor_filters)

        #print(gabor_processed.shape)
        
        #print(np.shape(np.array(gabor_filters)))
        # First competitive layer
        #Rate1, SynMat1 = competitive_layer(images, SynMat1, Learnrate, OutputSparseness, 0.0)
        
        #=====================================================
        # First competitive layer
        #Rate1= np.random.rand(32, 32)
        Rate1, SynMat1 = competitive_layer_2d(gabor_processed, Rate1, SynMat1, Learnrate_L1, OutputSparseness, alpha1, 1)
        

        print("Epoch "+str(epoch+1)+" Iteration " +str(iter + 1)+" The First Layer has been trained...")
        
        # Second competitive layer
        #Rate2= np.random.rand(32, 32)
        
        #print(Rate1*1000)
        #print(SynMat1)
        #======================================================
        
        Rate2, SynMat2 = competitive_layer_2d(Rate1, Rate2, SynMat2, Learnrate, OutputSparseness, alpha, 2)
        
        print("Epoch "+str(epoch+1)+" Iteration " +str(iter + 1)+" The Second Layer has been trained...")
        
        
        
        # Second competitive layer
        #Rate3= np.random.rand(32, 32)
        
        #=======================================================
        
        Rate3, SynMat3 = competitive_layer_2d(Rate2, Rate3, SynMat3, Learnrate, OutputSparseness, alpha, 3)
        
        print("Epoch "+str(epoch+1)+" Iteration " +str(iter + 1)+" The Third Layer has been trained...")
        
        
        
        # Second competitive layer
        #Rate4= np.random.rand(32, 32)
        
        #Rate4=Rate1.clone()
        
        Rate4, SynMat4 = competitive_layer_2d(Rate3, Rate4, SynMat4, Learnrate, OutputSparseness, alpha, 4)
        
        print("Epoch "+str(epoch+1)+" Iteration " +str(iter + 1)+" The Fourth Layer has been trained...")
        
        
        #=======================================================Global Learning
       
        # input_data = torch.tensor(gabor_processed, dtype=torch.float32)
        # SynMat1 = update_weights_2d_circular2_L1(SynMat1, Rate2, input_data, Rate2, Learnrate_L1, alpha1)
        # SynMat1 = update_weights_2d_circular2_L1(SynMat1, Rate3, input_data, Rate3, Learnrate_L1, alpha1)
        # SynMat1 = update_weights_2d_circular2_L1(SynMat1, Rate4, input_data, Rate4, Learnrate_L1, alpha1)
        
        
        
        
        # SynMat2 = update_weights_2d_circular2(SynMat2, Rate2, Rate1, Rate2, Learnrate, alpha)
        # SynMat2 = update_weights_2d_circular2(SynMat2, Rate3, Rate1, Rate3, Learnrate, alpha)
        
        
        
        # SynMat3 = update_weights_2d_circular2(SynMat3, Rate4, Rate3, Rate4, Learnrate, alpha)
        
        
        #=======================================================
        
        
        #=======================================================
        activations = Rate4.view(batch_size, layer_size[0]*layer_size[1])
        activations_tensor = torch.tensor(activations, dtype=torch.float32).to(device)
        labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=num_clases)
        
        
        all_outputs.append(activations_tensor)
        all_labels.append(labels_one_hot)
        
        #print(labels)
        
        if iter % 4000 == 9:  # Print every 10 mini-batches
            #activations = process_images_through_competitive_layers(images, gabor_filters, SynMat_layers)
            
            
            # Zero the parameter gradients
            #optimizer.zero_grad()
            
            
            # Concatenate the list elements into a single tensor
            all_outputs_tensor = torch.cat(all_outputs, dim=0).clone()
            all_labels_tensor = torch.cat(all_labels, dim=0).clone()

            lastw = compute_classification_weights(all_outputs_tensor, all_labels_tensor)
            #lastw = compute_classification_weights(activations_tensor, labels_one_hot)
            outputs = torch.matmul(lastw.T,activations_tensor.T)
            # Define a softmax layer
            # Apply softmax
            outputs = softmax_layer(outputs)
            #print(outputs.shape)
            #print(labels.shape)
            #print(lastw.shape)
            #print(activations_tensor.shape)
            #for i in range(10):
            #    # Forward + backward + optimize
            #    outputs = classification_layer(activations_tensor)
            #    #print(outputs.shape)
            #    #print(labels.shape)
            loss = criterion(outputs.T, labels)
            #    loss.backward()
            #    optimizer.step()
            # Print statistics
            #running_loss += loss.item()
            batch_accuracy = calculate_accuracy(outputs, labels)
            
            #running_accuracy += batch_accuracy
            
            print(f'===================================================================')
            print(f'[{epoch + 1}, {iter + 1}] loss: {loss.item():.3f}, batch accuracy: {batch_accuracy * 100:.2f}%')
            #print(f'[{epoch + 1}, {iter + 1}] Average Loss: {running_loss / (iter+1):.3f}, Averge Accuracy: {running_accuracy * 100/(iter+1):.2f}%')
            print(f'===================================================================')
            #running_loss = 0.0
            #running_accuracy = 0.0
            all_outputs=[]
            all_labels=[]



# Concatenate the list elements into a single tensor
all_outputs_tensor = torch.cat(all_outputs, dim=0)
all_labels_tensor = torch.cat(all_labels, dim=0)

lastw = compute_classification_weights(all_outputs_tensor, all_labels_tensor)




    # Test set evaluation
  # Assuming classification_layer is a PyTorch module adjusted for 2D input
classification_layer.eval()  # Set the network to evaluation mode

#OutputSparseness = 0.5
correct = 0
total = 0
#OutputSparseness = 0.01

with torch.no_grad():

    #for images, labels in testloader:
    for iter, data in enumerate(test_loader, 0):
        images, labels = data
        
        if iter>3000:
            break
            
        labels = labels.to(device)
        #images = torch.tensor(images, dtype=torch.float32)
        # Process each image through the 2D competitive layers with circular receptive fields
        # Assume 'process_images_through_competitive_layers_2d' is a function that handles 2D input properly,
        # accounting for the network's architecture, including circular receptive fields and sparseness-based inhibition.
        activations = process_images_through_competitive_layers_2d_circular(images, gabor_filters, OutputSparseness, SynMat1, SynMat2, SynMat3, SynMat4)
        
        # Depending on how 'activations' are structured, you might need to reshape or further process them 
        # to match the input requirements of your classification layer.
        activations_tensor = torch.tensor(activations, dtype=torch.float32).to(device)
        
        # Forward pass through the classification layer
        outputs = classification_layer(activations_tensor.view(activations_tensor.size(0), -1))  # Reshape if necessary
        
        labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=num_clases)
        #lastw = compute_classification_weights(activations_tensor, labels_one_hot)
        outputs = torch.matmul(lastw.T,activations_tensor.T).to(device)
        outputs = softmax_layer(outputs)
        
        
        print(outputs)
        print(labels)
        #print(outputs.shape)
        #print(labels.shape)
        #print(lastw.shape)
        #print(activations_tensor.shape)
        #for i in range(10):
        #    # Forward + backward + optimize
        #    outputs = classification_layer(activations_tensor)
        #    #print(outputs.shape)
        #    #print(labels.shape)
        loss = criterion(outputs.T, labels)
        #    loss.backward()
        #    optimizer.step()
        
        batch_accuracy = calculate_accuracy(outputs, labels)
        print(f'[{iter + 1}] loss: {loss.item():.3f}, batch test accuracy: {batch_accuracy * 100:.2f}%')
        
        # Get predictions and calculate accuracy
        _, predicted = torch.max(outputs, 0)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = correct / total
print(f'Test accuracy: {test_accuracy * 100:.2f}%')

print('Finished Training')






# for i in range(min(20, N)):  # Visualize the first 10 neurons
#     #plt.imshow(torch.tensor(SynMat1[:, i].reshape(layer_size[0], layer_size[1])).clone().detach(), cmap='gray')
#     plt.imshow(torch.tensor(gabor_processed[0, i, :, :]), cmap='viridis')
#     #plt.imshow(torch.tensor(all_mask[0, i*i, :].reshape(layer_size[0], layer_size[1])).cpu())
#     plt.title(f'Gabor Filter {i+1}')
#     plt.show()
    


for i in range(min(20, N)):  # Visualize the first 10 neurons
    #plt.imshow(torch.tensor(SynMat1[:, i].reshape(layer_size[0], layer_size[1])).clone().detach(), cmap='gray')
    plt.imshow(torch.tensor(gabor_processed[0, i, :, :]), cmap='viridis')
    #plt.imshow(torch.tensor(all_mask[0, i*i, :].reshape(layer_size[0], layer_size[1])).cpu())
    plt.title(f'Gabor Filter {i+1}')
    plt.show()
    
    

# Visualization or further evaluation can be added here.

# Evaluation: Visualize some synaptic weight matrices

print(SynMat2.shape)
print(max(SynMat2))

for i in range(0, min(10000, N), 100):  # Visualize the first 10 neurons
    #plt.imshow(torch.tensor(SynMat1[:, i].reshape(layer_size[0], layer_size[1])).clone().detach(), cmap='gray')
    plt.imshow(torch.tensor(SynMat2[0, :, i].T.reshape(
        layer_size[0], layer_size[1])).cpu()*10000000000.0, cmap='viridis')
    #plt.imshow(torch.tensor(all_mask[0, i*i, :].reshape(layer_size[0], layer_size[1])).cpu())
    plt.title(f'L2 Neuron {i+1}')
    plt.show()

for i in range(0, min(10000, N), 100):  # Visualize the first 10 neurons
    #plt.imshow(torch.tensor(SynMat1[:, i].reshape(layer_size[0], layer_size[1])).clone().detach(), cmap='gray')
    plt.imshow(torch.tensor(SynMat3[0, :, i].T.reshape(
        layer_size[0], layer_size[1])).cpu()*10000000000.0, cmap='viridis')
    #plt.imshow(torch.tensor(all_mask[0, i*i, :].reshape(layer_size[0], layer_size[1])).cpu())
    plt.title(f'L3 Neuron {i+1}')
    plt.show()

for i in range(0, min(10000, N), 100):  # Visualize the first 10 neurons
    #plt.imshow(torch.tensor(SynMat1[:, i].reshape(layer_size[0], layer_size[1])).clone().detach(), cmap='gray')
    plt.imshow(torch.tensor(SynMat4[0, :, i].T.reshape(
        layer_size[0], layer_size[1])).cpu()*10000000000.0, cmap='viridis')
    #plt.imshow(torch.tensor(all_mask[0, i*i, :].reshape(layer_size[0], layer_size[1])).cpu())
    plt.title(f'L4 Neuron {i+1}')
    plt.show()
