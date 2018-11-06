# Histogram Equalization

## A Simple Example
``` python
from IPL import IP
I = IP.ImageProcess()
I.readImage("img.png")
I_histEq = I.histogramEqualization()
I_histEq.showImage()
```
