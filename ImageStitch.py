from Stitcher import Stitcher
import cv2 as cv

imageA = cv.imread('moutains_1.png')
imageB = cv.imread('moutains_2.png')


# imageB = cv.flip(imageB, -1)
# imageA = cv.flip(imageA, -1)

sticher = Stitcher()
(result, vis) = sticher.stitch([imageA, imageB], showMatches=True)

# cv.imshow('imageA', imageA)
# cv.imshow('imageB', imageB)
# cv.imshow('vis', vis)
# cv.imshow('result', result)
# cv.waitKey(0)
# cv.destroyAllWindows()

cv.imwrite('sift_fla_vis.jpg', vis)
cv.imwrite('sift_fla_res.jpg', result)
