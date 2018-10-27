from PIL import Image
class ImageProcessColor:
	def __init__(self, image = None):
		self.im = image
		self.histogram = None

	def getImage(self):
		return self.im

	def readImage(self, filename):
		self.im = Image.open(filename)

	def showImage(self):
		self.im.show()

	def saveImage(self, filename):
		self.im.save(filename)

	def turnGray(self):
		if self.im == None:
			raise ValueError("No image")
		grayImg = self.im.convert('L')
		return ImageProcess(grayImg)
