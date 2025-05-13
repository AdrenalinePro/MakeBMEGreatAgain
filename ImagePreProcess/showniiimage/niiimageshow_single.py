import os
import nibabel as nb
import matplotlib.pyplot as plt
import numpy as np

t1=nb.load('C:/Users/23759/Desktop/SRTP/corrected_T2FSEAXs003a1001.nii.gz')
data=np.asanyarray(t1.dataobj)
plt.imshow(data[:, :, data.shape[2] // 2].T, cmap='Greys_r')
print(data.shape)
plt.show()