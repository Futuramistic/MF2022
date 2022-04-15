from graph_cut import GraphCut
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.stats as stats
from tqdm import TqdmSynchronisationWarning, tqdm

def normal(x,mean,sigma):
    return ( 2.*np.pi*sigma**2. )**-0.5 * np.exp( -0.5 * (x-mean)**2. / sigma**2. )

def task1_1():
    # Create gauss
    mu1, mu2 = 25.,100.
    sigma1, sigma2 = 5.,10.
    n = 10000
    x_gauss1 = np.random.normal(mu1,sigma1,n)
    y_gauss1 = normal(x_gauss1,mu1,sigma1)

    x_gauss2 = np.random.normal(mu2,sigma2,n)
    y_gauss2 = normal(x_gauss2,mu2,sigma2)

    # Stack
    gauss1 = np.vstack([x_gauss1,y_gauss1]).transpose()
    gauss2 = np.vstack([x_gauss2,y_gauss2]).transpose()
    # Sort
    gauss1 = np.array(sorted(gauss1,key=lambda row: row[0]))
    gauss2 = np.array(sorted(gauss2,key=lambda row: row[0]))

    wassertain = round((1./n)*np.sum(np.linalg.norm(gauss1-gauss2, axis=0)**2),2)
    m1 = np.array([mu1])
    m2 = np.array([mu2])
    closed_form = np.linalg.norm(m1-m2)**2 + (sigma1**2+sigma2**2-2*sigma2*sigma1)

    print(f"Closed-form Wassertain distance:    {closed_form}")
    print(f'Computed Wassertain distance:       {wassertain}')
    print(f'Deviation from actual:              {round(np.abs(wassertain-closed_form),2)}')
    print(f'Percentage of deviation:            {round(100*(np.abs(wassertain-closed_form)/closed_form),2)}')

    plt.figure(figsize=(30,30))
    sigmas = {0.25, 0.5, 0.75}
    for sigma in sigmas:
        plt.scatter(sigma*gauss1[:,0]+(1-sigma)*gauss2[:,0],sigma*gauss1[:,1]+(1-sigma)*gauss2[:,1])
    plt.scatter(gauss1[:,0],gauss1[:,1])
    plt.scatter(gauss2[:,0],gauss2[:,1])
    plt.show()

def task1_2(max_iter=100, n_vectors=5):

    source = np.array(mpimg.imread('code/OptimalTransport/src.png'))
    target = np.array(mpimg.imread('code/OptimalTransport/dst.png'))

    h = source.shape[0]
    w = source.shape[1]

    for iter in tqdm(range(max_iter)):
        updates = np.zeros(shape=(h*w,3), dtype=np.float64)
        for i in range(n_vectors):
            # Get random vector
            rand_vec = np.random.normal(size=3)
            rand_vec /= np.linalg.norm(rand_vec)

            # Project
            source_proj = source.dot(rand_vec).flatten()
            target_proj = target.dot(rand_vec).flatten()
            
            # Sort
            source_id = np.argsort(source_proj)
            target_id = np.argsort(target_proj)

            # U values -> for ith source pixel, target_{s_i} - source[i] is equal to update
            u = (target_proj[target_id] - source_proj[source_id]).reshape((h*w,1))
            updates[source_id] += u * rand_vec.transpose()
        source+=updates.reshape((h,w,3))/((np.float64)(n_vectors))
    
    fig, ax = plt.subplots(1, 3, figsize=(30, 30))
    images = [np.array(mpimg.imread('code/OptimalTransport/src.png')),target,source]
    titles = ["Source","Target","Result"]
    plt.tight_layout()
    for i,img in enumerate(images):
        ax[i].imshow(img)
        ax[i].set_title(titles[i])
        ax[i].axis('off')

    plt.show()
    pass

def main():
    # TODO: Implement Task 1
    task1_1()
    task1_2()


if __name__ == '__main__':
    main()
