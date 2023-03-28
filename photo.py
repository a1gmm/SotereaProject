import cv2
import numpy as np

if __name__ == "__main__":
    for n in range(1, 2):
        image = cv2.imread("%d.jpg"%(n))
        cv2.imwrite("%d.jpg" % n, image)
        #
        # height, width, _ = image.shape
        # print(f"height = {height}\nwidth = {width}\n")
        #
        # left_top_image = image[:height // 2, :width // 2, :]
        # #cv2.imwrite("%dleft_top_image.jpg" % n, left_top_image)
        #
        binary_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        #cv2.imwrite("%dbinary.jpg" % n, binary_image)
        #
        # print(f"shape = {image.shape}\n"
        #       f"binary shape = {binary_image.shape}\n")
        #
        for thresh in range(0, 101, 20):
             ret, thresh_image = cv2.threshold(binary_image, thresh, 255, cv2.THRESH_BINARY)
        #     #cv2.imwrite(f"%dthresh{thresh}.jpg" % n, thresh_image)
        #
        # clean_image = cv2.dilate(thresh_image, (500, 500), iterations=10)
        # _, clean_image = cv2.threshold(clean_image, thresh, 255, cv2.THRESH_BINARY)
        # clean_image = cv2.erode(thresh_image, (300, 300), iterations=1)
        # clean_image[:height // 3, :] = 0
        # cv2.imwrite(f"%dclean{thresh}.jpg" % n, clean_image)

        contours, thresh = cv2.threshold(binary_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, 2)
        cv2.drawContours(image, contours, -1, (0, 0, 255), 2)

        cv2.imshow('result', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()