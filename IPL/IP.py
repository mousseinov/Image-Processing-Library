from PIL import Image
import numpy as np
from math import floor
from math import ceil
from math import log
import matplotlib.pyplot as plt
import matplotlib as ml
from scipy import fftpack
from matplotlib.colors import LogNorm

class ImageProcess:
	def __init__(self, image = None):
		self.im = image
		self.histogram = None
		self.dft = None
		self.dct = None

	def getImage(self):
		return self.im

	def readImage(self, filename):
		self.im = Image.open(filename)

	def showImage(self):
		self.im.show()

	def saveImage(self, filename):
		self.im.save(filename)

	def makeHistogram(self):
		image = np.array(self.im)
		rows = len(image) #height
		cols = len(image[0]) #width
		count = [0]*256

		for row in range(rows):
			for col in range(cols):
				count[int(image[row][col])] += 1

		for i in range(len(count)):
			count[i] /= float(rows*cols)

		self.histogram = count


	def downSample(self, dimensions):
		image = np.array(self.im)
		currHeight = len(image)
		currWidth = len(image[0])

		downSampledHeight = dimensions[0]
		downSampledWidth = dimensions[1]

		heightRatio = currHeight/downSampledHeight
		widthRatio = currWidth/downSampledWidth

		downSampledImage = np.zeros(dimensions, dtype = int)

		for row in range(downSampledHeight):
			for col in range(downSampledWidth):
				downSampledImage[row][col] = image[heightRatio*row][widthRatio*col]


		return Image.fromarray(downSampledImage.astype('uint8'))

	def upSample(self, dimensions, inter = "bilinear"):
		if self.im == None:
			return
		image = np.array(self.im)
		currHeight = len(image)
		currWidth = len(image[0])

		upSampledHeight = dimensions[0]
		upSampledWidth = dimensions[1]
	
		heightRatio = upSampledHeight/currHeight
		widthRatio = upSampledWidth/currWidth

		upSampledImage = np.zeros(dimensions, dtype = int)
		for row in range(upSampledHeight):
			for col in range(upSampledWidth):
				if row%heightRatio == 0 and col%widthRatio == 0:
					upSampledImage[row][col] = image[row/heightRatio][col/widthRatio]
				elif inter == "NN" :
					Y = row/float(heightRatio)
					X = col/float(widthRatio)
					x1 = int(floor(X))
					y1 = int(floor(Y))
					upSampledImage[row][col] = image[y1%currHeight][x1%currWidth]
				else:
					Y = row/float(heightRatio)
					X = col/float(widthRatio)
					x1 = int(floor(X))
					x2 = int(ceil(X))
					y1 = int(floor(Y))
					y2 = int(ceil(Y))
					
					p = X%1
					q = Y%1
					
					A = int(image[y1%currHeight][x1%currWidth])
					B = int(image[y1%currHeight][x2%currWidth])
					C = int(image[y2%currHeight][x1%currWidth])
					D = int(image[y2%currHeight][x2%currWidth])

					f1 = (1-p)*A + p*B
					f2 = (1-p)*C + p*D

					f = (1-q)*f1 + q*f2
					
					upSampledImage[row][col] = int(f)%256


		return Image.fromarray(upSampledImage.astype('uint8'))

	def randomDither(self, bits):
		image = np.array(self.im)
		levels = 2**bits
		delta = 256/levels
		quantLevels = [i for i in range(0, 255, delta)]
		currHeight = len(image)
		currWidth = len(image[0])
		for row in range(currHeight):
			for col in range(currWidth): 
				decision = random.randint(0,3)
				if decision == 0:
					level = random.randint(0, len(quantLevels)-1)
					image[row][col] = quantLevels[level]

		return Image.fromarray(image.astype('uint8'))

	def histogramEqualization(self):
		self.makeHistogram()
		image = np.array(self.im)
		rows = height = len(image) #height
		cols = width = len(image[0]) #width

		dimensions = (height, width)
		equalizedImage = np.zeros(dimensions, dtype = int)
		for row in range(rows):
			for col in range(cols):
				eqPixel = 0
				k = image[row][col]
				pixVal = 0
				for n in range(k): pixVal += 255*self.histogram[n]
				equalizedImage[row][col] = floor(pixVal)%256
		return ImageProcess(Image.fromarray(equalizedImage.astype('uint8')))

	def plotEqualizationFunction(self):
		equalizationArray = [0]*255
		for n in range(0, 255):
			pixelVal = 0
			for i in range(n): pixelVal += 255*self.histogram[i]
			equalizationArray[n] = pixelVal

		plt.plot(equalizationArray)
		plt.ylabel('Intensity Equalized (u)')
		plt.xlabel('Intensity (v)')
		plt.title('u vs v python implementation')
		plt.savefig("uv")
		plt.show()

	def fft2D(self):
		if self.im == None:
			raise ValueError("No Image in object")
		image = np.array(self.im)
		shiftedfft = fftpack.fftshift(fftpack.fft2(image))
		self.dft = shiftedfft

	def dct2D(self):
		if self.im == None:
			raise ValueError("No Image in object")
		image = np.array(self.im)
		newImage = np.zeros((len(image),len(image[0])))
		for row in range(len(newImage)):
			for col in range(len(newImage[0])):
				newImage[row][col] = int(image[row][col])

		self.dct = fftpack.fftshift(fftpack.dct(fftpack.dct(newImage.T, norm='ortho').T, norm='ortho'))

	def dctImage(self, filename):
		plt.figure()
		plt.imshow(np.abs(self.dct), norm=LogNorm(vmin=5))
		plt.colorbar()
		title = ((filename.split("/"))[-1])
		plt.title(title)
		plt.savefig(filename)

	def fftMagnitudeImage(self, filename):
		plt.figure()
		plt.imshow(np.abs(self.dft), norm=LogNorm(vmin=5))
		plt.colorbar()
		title = ((filename.split("/"))[-1])
		plt.title(title)
		plt.savefig(filename)


	def dftTruncate(self, decrease = 0.25):
		self.fft2D()
		trunc = int(1/decrease)
		dftCopy = np.copy(self.dft)
		shape = self.dft.shape
		truncLoRows = ((trunc-1)*shape[0])//(trunc*2)
		truncHiRows = ((trunc+1)*shape[0])//(trunc*2)
		truncLoCols = ((trunc-1)*shape[1])//(trunc*2)
		truncHiCols = ((trunc+1)*shape[1])//(trunc*2)
		for row in range(shape[0]):
			for col in range(shape[1]):
				if row < truncLoRows or row > truncHiRows or col < truncLoCols or col > truncHiCols:
					dftCopy[row][col] = 0
			
		imreg = fftpack.ifftshift(dftCopy)
		imreg = fftpack.ifft2(imreg).real
		for i in range(len(imreg)):
			for j in range(len(imreg[0])):
				if imreg[i][j] < 0:
					imreg[i][j] = 0
				elif imreg[i][j] > 255:
					imreg[i][j] = 255
				else:
					imreg[i][j] = (floor(imreg[i][j]) + 1)%256
		return ImageProcess(Image.fromarray(imreg.astype('uint8')))

	def dctTruncate(self, decrease = 0.25):
		trunc = int(1/decrease)
		dctCopy = np.copy(self.dct)
		shape = self.dct.shape
		truncLoRows = ((trunc-1)*shape[0])//(trunc*2)
		truncHiRows = ((trunc+1)*shape[0])//(trunc*2)
		truncLoCols = ((trunc-1)*shape[1])//(trunc*2)
		truncHiCols = ((trunc+1)*shape[1])//(trunc*2)
		for row in range(shape[0]):
			for col in range(shape[1]):
				if row < truncLoRows or row > truncHiRows or col < truncLoCols or col > truncHiCols:
					dctCopy[row][col] = 0

		imreg = fftpack.ifftshift(dctCopy)
		reconImg = fftpack.idct(fftpack.idct(imreg.T, norm='ortho').T, norm='ortho')
		for i in range(len(reconImg)):
			for j in range(len(reconImg[0])):
				if reconImg[i][j] < 0:
					reconImg[i][j] = 0
				elif reconImg[i][j] > 255:
					reconImg[i][j] = 255
				else:
					reconImg[i][j] = (floor(reconImg[i][j]) + 1)%256
		return ImageProcess(Image.fromarray(reconImg.astype('uint8')))
	
		




