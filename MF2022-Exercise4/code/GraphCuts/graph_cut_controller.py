import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from tkinter import *
from PIL import Image

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
        

    # TODO: TASK 2.3
    # Hint: Use coo_matrix from the scipy.sparse library to initialize large matrices
    # The coo_matrix has the following syntax for initialization: coo_matrix((data, (row, col)), shape=(width, height))
    def __get_pairwise(self, image):
        """
        Get pairwise terms for each pairs of pixels on image
        :param image: color image as a numpy array
        :return: pairwise : sparse square matrix containing the pairwise costs for image
        """
        
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
        cost_fg = self.__get_color_histogram(image_array, seed_fg, hist_res)
        cost_bg = self.__get_color_histogram(image_array, seed_bg, hist_res)

        # TODO: TASK 2.2-2.3 - set the unaries and the pairwise terms
        unaries = self.__get_unaries(image_array, lambda_value, cost_fg, cost_bg, seed_fg, seed_bg)
        pairwise = self.__get_pairwise(image_array)

        # TODO: TASK 2.4 - perform graph cut
        # Your code here

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
