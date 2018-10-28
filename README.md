## Summary
A library of Image Processing tools that are wrapped around the `PIL` 
library in Python

## Features
- Downsampling
- Upsampling
- blurring an image using the Discrete Fourier Transform
- blurring an image using the Discrete Cosine Transform


## A Simple Example
``` python 
from ImageProcess import ImageProcess
I = ImageProcess()
I.readImage("img.png")
I_blurred = I.dftTruncate(0.25)
I_blurred.showImage()
```
https://user-images.githubusercontent.com/25520872/47620232-44efae80-dabe-11e8-93dc-e8b6ee6b8cfd.png
