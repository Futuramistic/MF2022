import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from tkinter import *
from PIL import Image
from tqdm import tqdm
import cv2

from graph_cut import GraphCut
from graph_cut_gui import GraphCutGui


class GraphCutController:

    def __init__(self):
        self.__init_view()

    def __init_view(self):
        root = Tk()
        root.geometry("700x500")
        self._view = GraphCutGui(self, root)
        root.mainloop()

    def __ToRGBImage(self,image):
        if len(image.shape) > 2 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        return image

    # TODO: TASK 2.1
    def __get_color_histogram(self, image, seed, hist_res):
        """
        Compute a color histograms based on selected points from an image
        
        :param image: color image
        :param seed: Nx2 matrix containing the the position of pixels which will be
                    used to compute the color histogram
        :param histRes: resolution of the histogram
        :return hist: color histogram
        """

        rgb_values = np.array(image[seed[:, 1], seed[:,0],:])
        hist, _ = np.histogramdd(rgb_values,hist_res, range=[(0,256),(0,256),(0,256)])
        smooth_hist = ndimage.gaussian_filter(hist, sigma=0.1)
        smooth_hist = smooth_hist / np.sum(smooth_hist)
        return smooth_hist

    # TODO: TASK 2.2
    # Hint: Set K very high using numpy's inf parameter
    def __get_unaries(self, image, lambda_param, hist_fg, hist_bg, seed_fg, seed_bg):
        """

        :param image: color image as a numpy array
        :param lambda_param: lamdba as set by the user
        :param hist_fg: foreground color histogram
        :param hist_bg: background color histogram
        :param seed_fg: pixels marked as foreground by the user
        :param seed_bg: pixels marked as background by the user
        :return: unaries : Nx2 numpy array containing the unary cost for every pixels in I (N = number of pixels in I)
        """
        h,w = image.shape[0],image.shape[1]
        step = 256./32.
        bin = np.floor(image/step).astype(int)

        unaries = np.zeros(shape=(h,w,2), dtype=np.float64)

        # Pixels
        unaries[:,:,0] = lambda_param * -np.log(hist_fg[bin[:,:,0],bin[:,:,1],bin[:,:,2]]+1e-10)
        unaries[:,:,1] = lambda_param * -np.log(hist_bg[bin[:,:,0],bin[:,:,1],bin[:,:,2]]+1e-10)

        # Foreground
        unaries[seed_fg[:,1],seed_fg[:,0],1] = np.inf
        unaries[seed_fg[:,1],seed_fg[:,0],0] = 0

        # Background
        unaries[seed_bg[:,1],seed_bg[:,0],1] = 0
        unaries[seed_bg[:,1],seed_bg[:,0],0] = np.inf

        return unaries.reshape((h*w,2))
    # TODO: TASK 2.3
    # Hint: Use coo_matrix from the scipy.sparse library to initialize large matrices
    # The coo_matrix has the following syntax for initialization: coo_matrix((data, (row, col)), shape=(width, height))
    def __get_pairwise(self, image):
        """
        Get pairwise terms for each pairs of pixels on image
        :param image: color image as a numpy array
        :return: pairwise : sparse square matrix containing the pairwise costs for image
        """
        sigma = 5.
        h,w = image.shape[0],image.shape[1]
        graph_indices = np.arange(h*w).reshape(h,w)
        pairwise = []
        for index in tqdm(range(h*w)):
            i = int(index / w)
            j = int(index % w)
            neighbours = np.array([
                [i-1,j-1], # u-l
                [i-1,j],   # u
                [i-1,j+1], # u-r

                [i,j-1],   # l
                [i,j+1],   # r

                [i+1,j-1], # d-l
                [i+1,j],   # d
                [i+1,j+1]  # d-r
            ])
            boundary =  np.logical_and(np.logical_and(0 <= neighbours[:,0],neighbours[:,0]<h),np.logical_and(0<=neighbours[:,1],neighbours[:,1]<w))
            neighbours=neighbours[boundary]
            neighbours_indices = graph_indices[neighbours[:,0],neighbours[:,1]]
            neighbours_rgb= image[neighbours[:,0],neighbours[:,1]].astype(np.float64)

            rgb_dist    =     np.linalg.norm(neighbours_rgb - image[i,j], axis=1)
            spat_dist   =     np.linalg.norm(neighbours - [i,j], axis=1)

            costs = np.exp(-(rgb_dist**2)/(2*sigma**2))/spat_dist
            for n in range(neighbours.shape[0]):
                pairwise.append([index, neighbours_indices[n], 0, costs[n], 0, 0])
        return pairwise
        
    # TODO TASK 2.4 get segmented image to the view
    def __get_segmented_image(self, image, labels, background=None):
        """
        Return a segmented image, as well as an image with new background 
        :param image: color image as a numpy array
        :param label: labels a numpy array
        :param background: color image as a numpy array
        :return image_segmented: image as a numpy array with red foreground, blue background
        :return image_with_background: image as a numpy array with changed background if any (None if not)
        """
        h,w = image.shape[0],image.shape[1]

        mask = np.zeros((h,w,3), dtype=np.uint8)
        # Red - foreground
        mask[~labels]  = np.array([255,0,0], dtype=np.uint8)
        # Blue - background
        mask[labels]   = np.array([0,0,255], dtype=np.uint8)

        # Segmented image - blend
        image_segmented = np.array(Image.blend(Image.fromarray(image),Image.fromarray(mask),alpha=0.5))
        image_with_background = None

        # Segmented image with background
        if background is not None:
            # Bound foreground
            foreground_in = np.where(labels==False)
            h_min,h_max = np.amin(foreground_in[0]),np.amax(foreground_in[0])
            w_min,w_max = np.amin(foreground_in[1]),np.amax(foreground_in[1])
            
            # Extract foreground
            foreground          = image[h_min:h_max,w_min:w_max]
            foreground_labels   = labels[h_min:h_max,w_min:w_max]

            f_h,f_w = foreground.shape[0],foreground.shape[1]
            b_h,b_w = background.shape[0],background.shape[1]

            if  b_h < f_h or b_w < f_w:
                print("Background image should be larger than extracted foreground!")
                return image_segmented,None
            
            # Offset image
            w_offset =  int((b_w-f_w)/2)  # Center width
            h_offset =  int((b_h-f_h))    # Place at the bottom

            mask = np.zeros((b_h, b_w), dtype=np.bool8)
            mask[h_offset:+f_h+h_offset,w_offset:f_w+w_offset] = (~foreground_labels).astype(np.bool8)

            image_with_background = np.copy(background)
            # PNG image -> drop channel
            image_with_background=self.__ToRGBImage(image_with_background)
            image_with_background[mask] = foreground[~foreground_labels]

        return image_segmented,image_with_background



    def segment_image(self, image, seed_fg, seed_bg, lambda_value, background=None):
        image_array = np.asarray(image)
        background_array = None
        if background:
            background_array = np.asarray(background)
        seed_fg = np.array(seed_fg)
        seed_bg = np.array(seed_bg)
        height, width = np.shape(image_array)[0:2]
        num_pixels = height * width

        # TODO: TASK 2.1 - get the color histogram for the unaries
        hist_res = 32
        # As sanity check, change image to RGB from possible RGBA
        image_array=self.__ToRGBImage(image_array)
        cost_fg = self.__get_color_histogram(image_array, seed_fg, hist_res)
        cost_bg = self.__get_color_histogram(image_array, seed_bg, hist_res)

        # TODO: TASK 2.2-2.3 - set the unaries and the pairwise terms
        unaries = self.__get_unaries(image_array, lambda_value, cost_fg, cost_bg, seed_fg, seed_bg)
        pairwise = self.__get_pairwise(image_array)

        # TODO: TASK 2.4 - perform graph cut
        # Your code here
        g = GraphCut(num_pixels, len(pairwise))
        g.set_unary(unaries)
        g.set_pairwise(pairwise)
        g.minimize()
        labels = g.get_labeling().reshape(height, width)

        # TODO TASK 2.4 get segmented image to the view
        segmented_image, segmented_image_with_background = self.__get_segmented_image(image_array, labels,
                                                                                      background_array)
        # transform image array to an rgb image
        segmented_image = Image.fromarray(segmented_image, 'RGB')
        self._view.set_canvas_image(segmented_image)
        if segmented_image_with_background is not None:
            segmented_image_with_background = Image.fromarray(segmented_image_with_background, 'RGB')
            plt.imshow(segmented_image_with_background)
            plt.show()
