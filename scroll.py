import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from collections import defaultdict


def display(image, duration=0):
    cv2.imshow("frame", image)
    key_press = cv2.waitKey(duration) & 0xFF

    if key_press == ord("q"):
        cv2.destroyAllWindows()


def load_video(video_path, undersample=30):

    video_object = cv2.VideoCapture(video_path)

    # Check if file opened successfully
    assert video_object.isOpened()

    # video meta-data
    fps = video_object.get(cv2.CAP_PROP_FPS)
    frame_count = int(video_object.get(cv2.CAP_PROP_FRAME_COUNT))
    # duration = frame_count/fps
    frame_width = int(video_object.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_object.get(cv2.CAP_PROP_FRAME_HEIGHT))

    buf_length = int(frame_count / undersample) + 1
    print(frame_count, buf_length)
    buf = np.empty((buf_length, frame_height, frame_width, 3), np.dtype("uint8"))

    ret = True
    i = 0

    for f_num in tqdm(range(0, frame_count)):
        ret, frame = video_object.read()
        if ret and f_num % undersample == 0:
            buf[i] = frame
            cv2.imwrite("./assets/test/images/{}.png".format(f_num), frame)
            # buf[i] = cv2.cvtColor(buf[i], cv2.COLOR_BGR2RGB)
            i += 1

    video_object.release()

    return buf


def binarize_image(image, binarize_threshold=50):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.threshold(img, binarize_threshold, 255, cv2.THRESH_BINARY)[1]

    return img


def get_connected_components(bin_img1, bin_img2, connected_threshold=25):

    diff = cv2.subtract(bin_img2, bin_img1)

    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        diff, None, None, None, 8, cv2.CV_32S
    )

    areas = stats[1:, cv2.CC_STAT_AREA]

    result = np.zeros((labels.shape), np.uint8)

    for i in range(0, nlabels - 1):
        if areas[i] >= connected_threshold:  # keep
            result[labels == i + 1] = 255

    return result


class ScrollDetector:
    def __init__(self, frame):
        # Initialise Tracker
        self.tracker = cv2.legacy.TrackerMOSSE_create()
        print(frame.shape[:2])
        (self.frame_height, self.frame_widht) = frame.shape[0:2]
        self.object_bbox = None  # (x,y,w,h)
        self.detected_bbox = None  # (x,y,w,h)
        self.object_tracking_status = False
        self.scroll_event = False

        # Thresholds
        self.scroll_threshold = 15

        self.initialize(frame)

    def get_largest_bbox(self, frame):
        bin_frame = binarize_image(frame, 50)
        kernel = np.ones((11, 11), np.uint8)
        dilated_mask = cv2.dilate(bin_frame, kernel, iterations=3)

        contours = cv2.findContours(
            dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = contours[0] if len(contours) == 2 else contours[1]

        x_max, y_max, w_max, h_max = 0, 0, 0, 0
        for cntr in contours:
            x, y, w, h = cv2.boundingRect(cntr)
            if w * h > w_max * h_max:
                x_max, y_max, w_max, h_max = x, y, w, h

        # cv2.rectangle(frame, (x_max, y_max), (x_max + w_max, y_max + h_max),
        # 				(0, 0, 255), 2)

        # display(frame)

        return (x_max, y_max, w_max, h_max)

    def update_bbox(self, frame):
        (x, y, w, h) = self.get_largest_bbox(frame)

        # update with a larger bbox for better tracking
        if w * h > self.object_bbox[2] * self.object_bbox[3]:
            self.object_bbox = (x, y, w, h)

    def reinitialize_tracker(self):
        self.tracker.clear()
        self.tracker = cv2.legacy.TrackerMOSSE_create()

    def initialize(self, frame):

        if self.object_bbox is None:
            self.object_bbox = self.get_largest_bbox(frame)

        if self.tracker:
            if self.object_bbox == (0, 0, 0, 0):
                self.object_bbox = None
            else:
                self.tracker.init(frame, self.object_bbox)

        # print ("Scroll detector initialized !!")

    def track_object(self, frame, show=True):
        if self.object_bbox is not None:

            (self.object_tracking_status, self.detected_bbox) = self.tracker.update(
                frame
            )

            # check to see if the tracking was a success
            if self.object_tracking_status:
                (x, y, w, h) = [int(v) for v in self.detected_bbox]
                if show:
                    # print ("Detected_box: ", x,y,w,h)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    display(frame)

            else:
                # print (self.detected_bbox)
                self.object_bbox = None

    def detect_scroll(self, frame, show=True):
        self.track_object(frame, show)
        # print (self.object_bbox)
        if self.object_bbox is not None:

            new_x_mid = int(self.detected_bbox[0] + (self.detected_bbox[2] / 2))
            new_y_mid = int(self.detected_bbox[1] + (self.detected_bbox[3] / 2))

            old_x_mid = int(self.object_bbox[0] + (self.object_bbox[2] / 2))
            old_y_mid = int(self.object_bbox[1] + (self.object_bbox[3] / 2))

            # scroll detected if midpoint shifts
            if (
                abs(new_x_mid - old_x_mid) > self.scroll_threshold
                or abs(new_y_mid - old_y_mid) > self.scroll_threshold
            ):

                self.object_bbox = self.detected_bbox
                self.scroll_event = True
                print("SCROLL DETECTED")
            else:

                self.scroll_event = False

        # object tracking lost due to previous scroll event
        elif self.object_bbox is None and self.scroll_event == True:
            self.scroll_event = False

            self.reinitialize_tracker()
            self.initialize(frame)

        # object tracking lost due to major changes in frame - counted as a scroll
        else:
            self.scroll_event = True
            self.reinitialize_tracker()
            self.initialize(frame)
            self.detected_bbox = self.object_bbox

        # print (self.scroll_event)


def extract_objects(files, binarize_threshold, connected_threshold):

    first_frame = cv2.imread("./assets/test/images/{}".format(files[0]))

    sd = ScrollDetector(first_frame)
    sd.detect_scroll(first_frame)

    for f in tqdm(range(0, len(files) - 1)):

        img1 = cv2.imread("./assets/test/images/{}".format(files[f]))
        img2 = cv2.imread("./assets/test/images/{}".format(files[f + 1]))

        bin_img1 = binarize_image(img1, binarize_threshold)
        bin_img2 = binarize_image(img2, binarize_threshold)

        result = get_connected_components(bin_img1, bin_img2, connected_threshold)

        sd.detect_scroll(img2)


if __name__ == "__main__":
    video_type = "test"
    video_name = "math"
    video_format = "mp4"
    video_path = "./assets/{}/{}.{}".format(video_type, video_name, video_format)
    cursor_template = cv2.imread("./assets/cursor.png", 0)
    binarize_threshold = 50  # darker colors need lower threshold (MAKE ADJUSTABLE)
    connected_threshold = 25  # removes video noise

    # load_video(video_path, 30)
    files = [i for i in os.listdir("./assets/test/images") if i.split(".")[-1] == "png"]
    # print (files)

    files = sorted(files, key=lambda x: int(x.split(".png")[0]))

    extract_objects(files, binarize_threshold, connected_threshold)
