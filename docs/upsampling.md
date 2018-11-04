# Upsampling

## Nearest Neighbor
`python
from IPL import IP
I = IP.ImageProcess()
I.readImage("img.png") # assume img.png 64 x 64
I_upsamp = I.upSample((128, 128), "NN" ) # upsample to 128 x 128

## Bilinear Interpolation
