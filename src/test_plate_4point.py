import cv2
import numpy as np

img = cv2.imread("./dataset/vehicle/frames/15/15_00041.jpg")

pts = [
    [100, 960],
    [140, 990],
    [85, 1000],
    [125, 1040]
]

cv2.namedWindow("Points", cv2.WINDOW_NORMAL)
cv2.imshow("Original", img)
cv2.waitKey(0)

# def order_points(pts):
#     rect = np.zeros((4, 2), dtype="float32")

#     s = pts.sum(axis=1)
#     rect[0] = pts[np.argmin(s)]
#     rect[2] = pts[np.argmax(s)]

#     diff = np.diff(pts, axis=1)
#     rect[1] = pts[np.argmin(diff)]
#     rect[3] = pts[np.argmax(diff)]

#     return rect


# def four_point_transform(image, pts):
#     pts = np.array(pts, dtype="float32")

#     rect = order_points(pts)
#     (tl, tr, br, bl) = rect

#     widthA = np.linalg.norm(br - bl)
#     widthB = np.linalg.norm(tr - tl)
#     maxWidth = int(max(widthA, widthB))

#     heightA = np.linalg.norm(tr - br)
#     heightB = np.linalg.norm(tl - bl)
#     maxHeight = int(max(heightA, heightB))

#     dst = np.array([
#         [0, 0],
#         [maxWidth-1, 0],
#         [maxWidth-1, maxHeight-1],
#         [0, maxHeight-1]
#     ], dtype="float32")

#     M = cv2.getPerspectiveTransform(rect, dst)
#     return cv2.warpPerspective(image, M, (maxWidth, maxHeight))


# warped = four_point_transform(img, pts)

# cv2.namedWindow("Points", cv2.WINDOW_NORMAL)
# cv2.imshow("Original", img)
# cv2.imshow("Warped", warped)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
