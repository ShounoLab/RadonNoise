'''
Radon Traonsform Examples
`lena.png` image (loaded as `rawimg`) transformed to `sngm`
In the case, the input image is masked with circle mask.
The `sngm` transoformed to a reconstructed iamge
'''
# -*- coding: utf-8 -*-
#

import numpy as np
from skimage.transform import radon, iradon
from skimage.io import imread
import matplotlib.pylab as plt


if __name__ == '__main__':
    fname = 'lena.png'
    rawimg = imread(fname)

    ysz, xsz = rawimg.shape[1], rawimg.shape[0]
    cy, cx = ysz/2, xsz/2
    XX, YY = np.meshgrid(np.arange(ysz)-cy, np.arange(xsz)-cx)
    cmask = XX**2 + YY**2 < (min(cy, cx)-5)**2
    img = cmask * rawimg
    thetas = np.linspace(0., 180., num=max(xsz, ysz)*5, endpoint=False)

    sngm = radon(img, theta=thetas, circle=True)
    recimg = iradon(sngm, theta=thetas, circle=True)

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('orig.')

    plt.subplot(2, 2, 2)
    plt.imshow(sngm, cmap='jet')
    plt.title('sinogram')

    plt.subplot(2, 2, 4)
    plt.imshow(recimg, cmap='gray')
    plt.title('recon.')

    plt.subplot(2, 2, 3)
    plt.imshow(recimg - img, cmap='jet')
    plt.title('res.')

    plt.show()
