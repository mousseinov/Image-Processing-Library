# Histogram Equalization
`function: histogramEqualization`
`parameters: None` <br/>
`returns: IP.ImageProcess object containing the equalized image`
## Example
``` python
from IPL import IP
I = IP.ImageProcess()
I.readImage("img.png")
I_histEq = I.histogramEqualization()
I_histEq.showImage()
```
