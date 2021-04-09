# Import python libraries
import numpy as np
import cv2

# set to 1 for pipeline images
debug = 0


class Segmentation(object):
    # Segmentation class to detect objects
    def __init__(self):
        self.fgbg = cv2.createBackgroundSubtractorMOG2()

    def detect(self, frame):
        '''Detect objects in video frame using following pipeline
            - Convert captured frame from BGR to GRAY
            - Perform Background Subtraction
            - Detect edges using Canny Edge Detection
              http://docs.opencv.org/trunk/da/d22/tutorial_py_canny.html
            - Retain only edges within the threshold
            - Dilation for objects
            - Find contours
            - Find centroids for each valid contours
        Args:
            frame: single image frame
        Return:
            centers: vector of object centroids in the source frame
        '''

        # Convert BGR to GRAY
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Perform Background Subtraction
        fgmask = self.fgbg.apply(gray)

        #cv2.imshow('bgsub', fgmask)

        # Detect edges
        edges = cv2.Canny(fgmask, 50, 190, 3)

        # Retain only edges within the threshold
        ret, thresh = cv2.threshold(edges, 127, 255, 0)
        cv2.imshow('thresh',thresh)
        
        # Dilate objects
        kernel = np.ones((5, 5), 'uint8')
        dilated = cv2.dilate(thresh, kernel, iterations=2)
        cv2.imshow('dilated',dilated)

        # Find contours
        contours, hierarchy = cv2.findContours(dilated,
                                                  cv2.RETR_EXTERNAL,
                                                  cv2.CHAIN_APPROX_SIMPLE)


        centers = []  # vector of object centroids in a frame
        # setting threshold size for cells
        blob_radius_thresh = 15
        # Find centroid for each valid contours
        for cnt in contours:
            try:
                # Calculate and draw circle
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                centeroid = (int(x), int(y))
                radius = int(radius)
                if (radius > blob_radius_thresh):
                    cv2.circle(frame, centeroid, radius, (0, 255, 0), 2)
                    b = np.array([[x], [y]])
                    centers.append(np.round(b))
            except ZeroDivisionError:
                pass

        # Show contours of tracking objects
        cv2.imshow('Track Cells', frame)

        return centers
