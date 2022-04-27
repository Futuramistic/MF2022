
import numpy as np 
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.sparse as spar
import scipy.sparse.linalg as lin

def compute_errors(original, images):
    errors = [np.abs(original - image).mean() for image in images]
    relative_errors = [(error / errors[0]) for error in errors]

    return relative_errors

def gaussian_kernel(sigma=0.5):
    length = (2.*np.ceil(3.*sigma)+1.).astype(np.int32)
    ax = np.linspace(-(length-1)/2.,(length-1)/2.,length)
    gauss = np.exp(-0.5 * ax**2/ sigma**2)
    kernel = np.outer(gauss,gauss)
    kernel /= np.sum(kernel)
    return kernel

def laplace_kernel():
    return np.array(
        [
            [0.,1.,0.],
            [1.,-4.,1.],
            [0.,1.,0.]
        ]
    ).astype(np.float64)

def convol(image, kernel, padding = 'edge'):
    ker_h, ker_w = kernel.shape[0], kernel.shape[1]

    pad_h = int((ker_h-1)/2)
    pad_w = int((ker_w-1)/2)

    if padding == 'edge':
        pad_img = np.pad(image,(pad_h,pad_w),'edge').astype(np.longdouble)
    elif padding == 'zero':
        pad_img = np.pad(image,(pad_h,pad_w),'constant').astype(np.longdouble)
    sub_shape = kernel.shape + tuple(np.subtract(pad_img.shape,kernel.shape)+1)
    sub_matrices = np.lib.stride_tricks.as_strided(pad_img,shape=sub_shape,strides=pad_img.strides*2,subok=True,writeable=False).astype(np.longdouble)
    return (np.einsum('ij,ijkl->kl',kernel,sub_matrices).astype(np.longdouble))

def heat_diffusion(image,org_img, iter=100, heat_step=0.1, padding='edge'):
    out = image.copy().astype(np.float64)
    log = []
    for it in tqdm(np.arange(iter)):
        log.append(out)
        out = out + heat_step*convol(out,laplace_kernel(),padding=padding)
    log.append(out)
    to_image(out).show()
    plot_steps(log,int(iter/4),"Diffusion")
    show_error(org_img,log,"iterations")
    return out,log

def gaussian_filter(image,org_img, iter=32, padding='edge'):
    out = image.copy().astype(np.float64)
    log = []
    for it in tqdm(np.arange(iter)):
        log.append(out)
        out = convol(out, gaussian_kernel(), padding=padding)
    log.append(out)
    to_image(out).show()
    show_error(org_img,log,"filtering")
    plot_steps(log,int(iter/4),"Filtered")
    return out,log

def variational_method(img):
    lam = 2.
    h,w = img.shape[0],img.shape[1]
    n = h*w
    A = spar.diags([-lam,-lam,(1+4*lam),-lam,-lam,], offsets=[-w,-1,0,1,w], shape=(n,n))
    out = lin.spsolve(A.tocsc().astype(np.float64),img.astype(np.float64).flatten()).reshape(h,w)
    to_image(out).show()
    return out

def to_image(img):
    return Image.fromarray((np.clip(img*255.0, 0, 255)).astype(np.uint8))

def show_error(org_img,log,title):
    errors = compute_errors(org_img, log)
    plt.plot(np.arange(len(log)), errors)
    plt.title("Evolution of the errors over the number of "+title)
    plt.xlabel("Number of "+title)
    plt.ylabel("Errors")
    plt.show()

def plot_steps(log, step_size, title):
    images = int((len(log)-1)/step_size)
    fig, ax = plt.subplots(int(images/2),2, figsize=(30,30))
    ax = ax.flatten()
    for i in range(images):
        ax[i].imshow(to_image(log[(i+1)*step_size]), cmap='gray')
        ax[i].axis('off')
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
    Image.fromarray((np.clip(I_n*255.0, 0, 255)).astype(np.uint8)).show()

    gaussian_img, gaussian_log = gaussian_filter(I_n,I_orig)
    heat_img, heat_log = heat_diffusion(I_n,I_orig)
    variational_method(I_n)

if __name__ == "__main__":
    main()