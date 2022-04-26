import numpy as np 
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt


def compute_errors(original, images):
    errors = [np.abs(original - image).mean() for image in images]
    relative_errors = [(error / errors[0]) for error in errors]

    return relative_errors

def gaussian_kernel(sigma=0.5):
    length =(2*np.ceil(3*sigma)+1).astype(np.int32)
    ax = np.linspace(-(length-1)/2,(length-1)/2,length)
    gauss = np.exp(-0.5 * np.square(ax)/ np.square(sigma))
    kernel = np.outer(gauss,gauss)
    kernel /=np.sum(kernel)
    return kernel

def laplace_kernel():
    return np.array(
        [[0,1,0],
        [1,-4,1],
        [0,1,0]]
    ).astype(np.float64)

def convol(image, kernel, padding = 'edge'):
    img_h, img_w = image.shape[0],  image.shape[1]
    ker_h, ker_w = kernel.shape[0], kernel.shape[1]
    out = np.zeros_like(image).astype(np.float64)

    pad_h = int((ker_h-1)/2)
    pad_w = int((ker_w-1)/2)

    if padding == 'edge':
        pad_img = np.pad(image,(pad_h,pad_w),'edge')
    else:
        pad_img = np.pad(image,(pad_h,pad_w),'constant')

    for h in range(img_h):
        for w in range(img_w):
            out[h,w] = np.sum(kernel * pad_img[h:h+ker_h,w:w+ker_w]).astype(np.float64)
    return out
    
def heat_diffusion(image, iter=100, heat_step=0.1, log_step = 25):
    out = image.copy().astype(np.float64)
    log = []
    for it in tqdm(np.arange(iter)):
        if(it%log_step == 0):
            log.append(out)
        out = out + heat_step*convol(out,laplace_kernel())
    return out,log

def gaussian_filter(image, iter=32, log_step=8):
    out = image.copy().astype(np.float64)
    log = []
    for it in tqdm(np.arange(iter)):
        if(it%log_step == 0):
            log.append(out)
        out = convol(out, gaussian_kernel())
    return out,log

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

    gaussian_img, gaussian_log = gaussian_filter(I_n)
    Image.fromarray((np.clip(gaussian_img*255.0, 0, 255)).astype(np.uint8)).show()

    heat_img, heat_log = heat_diffusion(I_n)
    Image.fromarray((np.clip(heat_img*255.0, 0, 255)).astype(np.uint8)).show()

    # Dummy error visualization
    ratios = np.linspace(0, 1, 10)
    errors = compute_errors(I_orig, [(1 - r) * I_n + r * I_orig for r in ratios])
    plt.plot(ratios, errors)
    plt.show()


if __name__ == "__main__":
    main()