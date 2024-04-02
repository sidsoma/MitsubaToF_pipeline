import OpenEXR, Imath
import re
from copy import deepcopy
import numpy as np 
# from PIL import Image
from scipy.io import savemat
import sys
import imageio
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image
plt.ioff()


inputFile = OpenEXR.InputFile(sys.argv[1])
pixelType = Imath.PixelType(Imath.PixelType.HALF)
dataWin = inputFile.header()['dataWindow']
imgSize = (dataWin.max.x - dataWin.min.x + 1, dataWin.max.y - dataWin.min.y + 1)
tmp = list(inputFile.header()['channels'].keys())

if(len(tmp) != 3):
    prog = re.compile(r"\d+")
    channels = np.array(np.argsort([int(re.match(prog, x).group(0)) for x in tmp], -1, 'stable'))
    channels[0::3], channels[2::3] = deepcopy(channels[2::3]),deepcopy(channels[0::3])
    tmp = np.array(tmp)
    tmp = tmp[list(channels)]
else:
    tmp = np.array(tmp)
    tmp[0], tmp[2] = tmp[2], tmp[0]

video = inputFile.channels(tmp, pixelType)
video = [np.reshape(np.frombuffer(video[i], dtype=np.float16), imgSize) for i in range(len(video))]
video = np.stack(video, axis=2)
# Reshape to separate R,G,B in final channel
video = np.stack([video[...,0::3],
                  video[...,1::3],
                  video[...,2::3]], axis=-2)
image = video.sum(-1)**(1/2.4)
# TODO: Replace saving mat by exr. EXR is more efficient
# Save video
imageio.mimwrite(sys.argv[2]+'.mp4', 
                 [(np.clip((video[...,k]*32)**(1/2.4),0,1)*255).astype('uint8')
                    for k in range(video.shape[-1])], 
                 fps=5)
# Save steady state image
filename = sys.argv[2]+'_steady.png'
image = (image - np.min(image)) / (np.max(image) - np.min(image))
matplotlib.image.imsave(filename, image)


# print(np.max(image))
# print(np.min(image))
# imageio.imwrite(,image)
# im = Image.fromarray(image)
# im.save(sys.argv[2]+'_steady.png')


# Save tof
tof_idx = np.argmax(video,axis=-1)
tof_idx_8bit = ((tof_idx/tof_idx.max())*255).astype('uint8')
import cv2
tof_heatmap = cv2.applyColorMap(tof_idx_8bit, cv2.COLORMAP_JET)
cv2.imwrite(sys.argv[2]+'_tof_idx.png',tof_heatmap)

print(video.shape)

save_dict = {'I': video}
np.savez(sys.argv[2], **save_dict)
