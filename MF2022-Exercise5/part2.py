
from configparser import Interpolation
import enum
from typing import List
import numpy as np 
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.sparse as spar
import scipy.sparse.linalg as lin
from enum import Enum

class Padding(Enum):
    EDGE = 0    # Neumann boundary condition (default) -> no graddient if edge values the same
    CONST = 1   # Dirichlet boundary condition -> zero padding by default in numoy

def compute_errors(original, images):
    errors = [np.abs(original - image).mean() for image in images]
    relative_errors = [(error / errors[0]) for error in errors]

    return relative_errors

# Gaussian kernel computation
# Return: (np.array) gaussian kernel
def gaussian_kernel(sigma=0.5):
    length = (2.*np.ceil(3.*sigma)+1.).astype(np.int32)
    ax = np.linspace(-(length-1)/2.,(length-1)/2.,length)
    gauss = np.exp(-0.5 * ax**2/ sigma**2)
    kernel = np.outer(gauss,gauss)
    kernel /= np.sum(kernel)
    return kernel.astype(np.float64)

# Laplace kernel computation
# Return: (np.array) laplace kernel
def laplace_kernel():
    return np.array(
        [
            [0.,1.,0.],
            [1.,-4.,1.],
            [0.,1.,0.]
        ]
    ).astype(np.float64)

# 2D convolution
# Inputs: 
# - image: (np.array) -     image to convolve
# - kernel: (np.array) -    kernel to use for convolution
# - padding: (Padding) -    padding used (Default: EDGE)
# Return: (np.array) image after convolution
def convol(image: np.array, kernel: np.array, padding: Padding = Padding.EDGE):
    ker_h, ker_w = kernel.shape[0], kernel.shape[1]

    pad_h = int((ker_h-1)/2)
    pad_w = int((ker_w-1)/2)

    #Padding
    if padding == Padding.EDGE:
        pad_img = np.pad(image,(pad_h,pad_w),'edge').astype(np.longdouble)
    elif padding == Padding.CONST:
        pad_img = np.pad(image,(pad_h,pad_w),'constant').astype(np.longdouble)
    # Default
    else:
        pad_img = np.pad(image,(pad_h,pad_w),'edge').astype(np.longdouble)

    # Vectorized; however - there was a slight variation in results in comparison to iterative
    # All in all - the variation accounted to an error of 1e-11 error for whole image (no pixel color change), but gained a significant speed up
    stride_shape = kernel.shape + tuple(np.subtract(pad_img.shape,kernel.shape)+1)
    stride_mat = np.lib.stride_tricks.as_strided(pad_img,shape=stride_shape,strides=pad_img.strides*2,subok=True,writeable=False).astype(np.longdouble)
    return (np.einsum('ij,ijkl->kl',kernel,stride_mat).astype(np.longdouble))

# Heat diffusion
# Inputs: 
# - image: (np.array)       -     image to diffuse
# - org_image: (np.array)   -     image with no noise
# - iter: int               -     number of iterations (Default: 100; should be even)
# - heat: float             -     heat step (Default: 0.1)
# - padding: (Padding)      -     padding used (Default: EDGE)
# Return: (np.array) image
def heat_diffusion(image:np.array,org_img:np.array, iter:int=100, heat_step:float=1e-1, padding: Padding = Padding.EDGE):

    # Computation
    out = image.copy().astype(np.float64)
    log = []
    for it in tqdm(np.arange(iter)):
        log.append(out)
        out = out + heat_step*convol(out,laplace_kernel(),padding=padding)
    log.append(out)
    print(f'Final error: {compute_errors(org_img,log)[iter]}')
    _, ax = plt.subplots(1,1, figsize=(30,30))
    ax.imshow(np.clip(out*255.0, 0.0, 255.0).astype(np.uint8), cmap='gray', vmin=0.0, vmax=255.0)
    ax.set_title("Result")
    ax.axis('off')
    plt.tight_layout()
    plt.show()
    to_image(out).show()

    # Statistics
    plot_steps(log,int(iter/4),"Diffusion")
    show_error(org_img,log,"iterations")
    return out

# Heat diffusion
# Inputs: 
# - image: (np.array)       -     image to filter
# - org_image: (np.array)   -     image with no noise
# - iter: int               -     number of iterations (Default: 32; should be even)
# - sigma: float            -     sigma for gaussian kernel (Default: 0.5)
# - padding: (Padding)      -     padding used (Default: EDGE)
# Return: (np.array) image
def gaussian_filter(image:np.array,org_img:np.array, iter:int=32, sigma: float = 0.5, padding=Padding.EDGE):
    
    # Computation
    out = image.copy().astype(np.float64)
    log = []
    for it in tqdm(np.arange(iter)):
        log.append(out)
        next = convol(out, gaussian_kernel(sigma=sigma), padding=padding)
        out = next.copy()
    log.append(out)
    print(f'Final error: {compute_errors(org_img,log)[iter]}')
    _, ax = plt.subplots(1,1, figsize=(30,30))
    ax.imshow(np.clip(out*255.0, 0.0, 255.0).astype(np.uint8), cmap='gray', vmin=0.0, vmax=255.0)
    ax.set_title("Result")
    ax.axis('off')
    plt.tight_layout()
    plt.show()
    to_image(out).show()

    # Statistics
    show_error(org_img,log,"filtering")
    plot_steps(log,int(iter/4),"Filtered")
    return out

# Variational method
# Inputs:
# - image: (np.array)       -   image with noise 
# - org_image: (np.array)   -   image with no noise
# - lam: (float)            -   lambda to use (default: 2.0)
# Return: (np.array) image
def variational_method(img,org_image,lam = 2.):
    h,w = img.shape[0],img.shape[1]
    n = h*w
    A = spar.diags([-lam,-lam,(1+4*lam),-lam,-lam,], offsets=[-w,-1,0,1,w], shape=(n,n))
    out = lin.spsolve(A.tocsc().astype(np.float64),img.astype(np.float64).flatten()).reshape(h,w)
    _, ax = plt.subplots(1,1, figsize=(30,30))
    ax.imshow(np.clip(out*255.0, 0.0, 255.0).astype(np.uint8), cmap='gray', vmin=0.0, vmax=255.0)
    ax.set_title("Result")
    ax.axis('off')
    plt.tight_layout()
    plt.show()
    to_image(out).show()
    print(f'Error: {compute_errors(org_image,[img,out])[1]}')
    del A
    return out

# To grayscale image
# Input: 
# - img (np.array): image with values [0,1]
# Return: Image in grayscale
def to_image(img):
    return Image.fromarray((np.clip(img*255.0, 0.0, 255.0)).astype(np.uint8))

# Show error
# Input:
# - org_img: (np.array) original image
# - log: (List[np.array]) log of images after ith iteration
# - title: str - title of graph
def show_error(org_img,log,title):
    errors = compute_errors(org_img, log)
    plt.plot(np.arange(len(log)), errors, marker="o", markersize=1, mfc ="r", mec="r")
    plt.title("Evolution of the errors over the number of "+title)
    plt.xlabel("Number of "+title)
    plt.ylabel("Errors")
    plt.show()

# Show steps of operation
# Input:
# - log: (List[np.array]) log of images after ith iteration
# - step_size: (int) how often to report
# - title: str - title of graph
def plot_steps(log:List, step_size:int, title:str=None):
    images = int((len(log)-1)/step_size)
    _, ax = plt.subplots(int(images/2),2, figsize=(30,30))
    ax = ax.flatten()
    for i in range(images):
        ax[i].imshow(np.clip(log[(i+1)*step_size]*255.0, 0.0, 255.0).astype(np.uint8), cmap='gray', vmin=0.0, vmax=255.0)
        ax[i].axis('off')
        if title is not None:
            ax[i].set_title(title+f" at time {(i+1)*step_size}")
    plt.show()

def main():

    I_orig_img = Image.open('lotr.jpg')
    I_orig_img = I_orig_img.convert('L')
    I_orig = np.array(I_orig_img)/255.0

    h, w = np.shape(I_orig)
    
    # Add noise
    gauss = np.random.normal(0, 0.22, (h, w))
    gauss = gauss.reshape(h, w)
    I_n = I_orig + gauss

    # Visualize
    I_orig_img.show()
    to_image(I_n).show()

    gaussian_img = gaussian_filter(I_n,I_orig)
    heat_img = heat_diffusion(I_n,I_orig)
    var_img = variational_method(I_n,I_orig)

if __name__ == "__main__":
    main()