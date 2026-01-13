# Histogram calculation example (c2_histogram1.py)

# Relevant library declaration
import numpy as np
import cv2
from matplotlib import pyplot as plt
from imgRead import imgRead
from createFolder import createFolder
#%%
# read video
img1 = imgRead("./images/img5.jpg", cv2.IMREAD_GRAYSCALE, 320, 240)
img1
displays = [("img5", img1)]
for (name, out) in displays:
    cv2.imshow(name, out)
    
cv2.waitKey(0)
cv2.destroyAllWindows()    
#%%        

# Histogram calculation
ch1 = [0]; ch2 = [0]; ch3 = [0]
ranges1 = [0, 256]; 
ranges2 = [0, 128]; # red
ranges3 = [128, 256] # abstract
histSize1 = [256]; histSize2 = [128]; histSize3 = [128]
# Calculate the total number of bins for a specific range of values ​​in a channel
hist1 = cv2.calcHist([img1], ch1, None, histSize1, ranges1) 
hist2 = cv2.calcHist([img1], ch2, None, histSize2, ranges2)
hist3 = cv2.calcHist([img1], ch3, None, histSize3, ranges3)

# Histogram output and storage
bin_x1 = np.arange(256)
bin_x2 = np.arange(128)
bin_x3 = np.arange(128) + 128
# graphing techniques
plt.title("Histogram")
plt.xlabel("Bin")
plt.ylabel("Frequency")
plt.plot(bin_x1, hist1, color='b') # line chart
plt.bar(bin_x2, hist2[:,0], width=6,color='r') # Bar Chart: Green
plt.bar(bin_x3, hist3[:,0], width=6,color='g') # Bar chart: red
plt.grid(True, lw = 1, ls = '--', c= '.75') 
plt.xlim([0,255])

# save video
save_dir = './code_res_imgs/c2_histogram1'
createFolder(save_dir)
plt.savefig(save_dir + "/" + "hist.png")

plt.show()