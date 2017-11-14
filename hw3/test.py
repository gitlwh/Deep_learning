#just test
from PIL import Image
import scipy.misc
import matplotlib.pyplot as plt
import numpy as np
import json

img = Image.open("./123.jpg")
print(img)
img = scipy.misc.imread("./123.jpg")
img=scipy.misc.imresize(img,[32,32,3])
data = json.load(open("./123.json"))
print(data)
print(data['meta']['clinical']["benign_malignant"])