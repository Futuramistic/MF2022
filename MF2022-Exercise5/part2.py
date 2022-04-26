import numpy as np 
from PIL import Image
import matplotlib.pyplot as plt


def compute_errors(original, images):
    errors = [np.abs(original - image).mean() for image in images]
    relative_errors = [(error / errors[0]) for error in errors]

    return relative_errors


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

    # Dummy error visualization
    ratios = np.linspace(0, 1, 10)
    errors = compute_errors(I_orig, [(1 - r) * I_n + r * I_orig for r in ratios])
    plt.plot(ratios, errors)
    plt.show()


if __name__ == "__main__":
    main()