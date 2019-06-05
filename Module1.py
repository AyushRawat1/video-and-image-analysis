from skimage.measure import compare_ssim
#it compares the two images with the function compare_ssim
#used for parsing the images
import imutils
#contains all the plotting and drawing tools
import cv2


imageA = cv2.imread('hoard.jpg')
#reading the image from the source
imageB = cv2.imread('hoard1.jpg')

grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
#grayscaling the image
grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

(score, diff) = compare_ssim(grayA, grayB, full=True)
#calculating the score and the differences in the images
diff = (diff * 255).astype("uint8")

# threshold the difference image, followed by finding contours to
# obtain the regions of the two input images that differ
thresh = cv2.threshold(diff, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

for c in cnts:
	# compute the bounding box of the contour and then draw the
	# bounding box on both input images to represent where the two
	# images differ
	(x, y, w, h) = cv2.boundingRect(c)
	cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)
 
# show the output images
cv2.imshow("Original", imageA)
cv2.imshow("Modified", imageB)
cv2.imshow("Diff", diff)
#cv2.imshow("Thresh", thresh)
cv2.waitKey(0)
#also we are printing the percentage degraded out of 100%
print("Percentage Degraded: {}".format((100-round(score*100)))+"%")
cv2.destroyAllWindows()
#destroying all the windows

def mouse_handler(event, x, y, flags, data) :
    
    if event == cv2.EVENT_LBUTTONDOWN :
        cv2.circle(data['im'], (x,y),3, (0,0,255), 5, 16);
        cv2.imshow("Image", data['im']);
        if len(data['points']) < 4 :
            data['points'].append([x,y])

def get_four_points(im):
    #detecting the 4 points of the inputs
    data = {}
    data['im'] = im.copy()
    data['points'] = []
    
    cv2.imshow("Image",im)
    cv2.setMouseCallback("Image", mouse_handler, data)
    cv2.waitKey(0)
    
    points = np.vstack(data['points']).astype(float)
    
    return points

if __name__ == '__main__' :

    im_src = cv2.imread('download.jpeg')

    size = (900,400,3)
    #defining the size of the image
    im_dst = np.zeros(size, np.uint8)

    
    pts_dst = np.array([ [0,0], [size[0] - 1, 0], [size[0] - 1, size[1] -1], [0, size[1] - 1 ]], dtype=float)
    
    
    cv2.imshow("Image", im_src)
    
    #getting the four points of the images that we input 
    pts_src = get_four_points(im_src);
    
    h, status = cv2.findHomography(pts_src, pts_dst)
    #applying homography to the images
      
    im_dst = cv2.warpPerspective(im_src, h, size[0:2])
    #changing the perspective of the image to the desired input of the mouse clicks
    
    cv2.putText(im_dst, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),(10, im_dst.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 1)
    
    cv2.imshow("Image", im_dst)
    #displaying the image with the perspective correction 
    cv2.waitKey(0)
    
    cv2.destroyAllWindows()
    #destroying all the windows