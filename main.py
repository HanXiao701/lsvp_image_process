import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import shutil
from collections import defaultdict
from scroll import ScrollDetector
import json


def display(
    image, duration=0
):  # if duration zero; press any key to move through frames
    cv2.imshow("frame", image)
    key_press = cv2.waitKey(duration) & 0xFF


def load_video(video_name, video_type, undersample=None):

    video_path = os.path.join(
        "assets", "videos", "{}.{}".format(video_name, video_type)
    )
    print(video_path)
    video = cv2.VideoCapture(video_path)

    # Check if file opened successfully
    assert video.isOpened()

    # video meta-data
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if undersample == None:
        undersample = round(fps)

    print(
        "Video details: {} fps, {} frames, {}x{} (w,h)\n".format(
            fps, frame_count, frame_width, frame_height
        )
    )
    print("Processing video....")
    print("Undersample rate:", undersample)

    if os.path.exists(os.path.join("assets", "processed", "images")):
        shutil.rmtree(os.path.join("assets", "processed", "images"))

    os.mkdir(os.path.join("assets", "processed", "images"))

    ret = True
    for f_num in tqdm(range(0, frame_count)):
        ret, frame = video.read()
        if ret and f_num % undersample == 0:

            cv2.imwrite(
                os.path.join("assets", "processed", "images", "{}.png".format(f_num)),
                frame,
            )

    video.release()

    # save video meta data
    meta_data = {
        "video": "{}.{}".format(video_name, video_type),
        "fps": fps,
        "total_frames": frame_count,
        "frame_width": frame_width,
        "frame_height": frame_height,
        "sample_rate": undersample,
    }
    with open(
        os.path.join("assets", "json", "{}_meta.json".format(video_name)), "w"
    ) as fp:
        json.dump(meta_data, fp, indent=1)


def detect_cursor(image, cursor_template):

    w, h = cursor_template.shape[::-1]

    res = cv2.matchTemplate(image, cursor_template, cv2.TM_CCOEFF_NORMED)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    bottom_right = (max_loc[0] + w, max_loc[1] + h)

    # higlight cursor with rectangle
    # cv2.rectangle(image, max_loc, bottom_right, 0, -1)

    cursor_pos = (max_loc, bottom_right)

    return cursor_pos


def remove_cursor(image, cursor_template):
    bin_image = binarize_image(image)

    cursor_pos = detect_cursor(bin_image, cursor_template)
    cv2.rectangle(image, cursor_pos[0], cursor_pos[1], 0, -1)

    return image


def binarize_image(image, binarize_threshold=50):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.threshold(img, binarize_threshold, 255, cv2.THRESH_BINARY)[1]

    return img


def compute_image_difference(img2, img1, binarize=True):

    if binarize:
        img2 = binarize_image(img2, binarize_threshold)

        img1 = binarize_image(img1, binarize_threshold)

    difference = cv2.subtract(img2, img1)

    return difference


def make_transparent(img):

    image_copy = img.copy()
    binary_image = binarize_image(image_copy)

    result = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    result[:, :, 3] = binary_image

    cv2.imwrite("result.png", result)


def get_connected_components(img, connected_threshold=25):

    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        img, None, None, None, 8, cv2.CV_32S
    )

    areas = stats[1:, cv2.CC_STAT_AREA]

    result = np.zeros((labels.shape), np.uint8)

    for i in range(0, nlabels - 1):
        if areas[i] >= connected_threshold:  # keep
            result[labels == i + 1] = 255

    return result


def compute_accuracy(predicted, ground_truth):

    detection = json.load(open(predicted))
    y_pred = [int(detection[i]["start"]) for i in detection]

    y_true = []
    with open(ground_truth) as file:
        for line in file:
            y_true.append(int(line.rstrip()))

    print("True: ", y_true)
    print("Predicted: ", y_pred)

    TP = []
    FP = y_pred.copy()
    FN = y_true.copy()

    for i in range(0, len(y_pred)):
        for j in range(0, len(y_true)):
            if (
                abs(y_pred[i] - y_true[j]) <= 30
            ):  # count 1 second difference as detected
                if y_true[j] in FN:
                    TP.append(y_pred[i])
                    FP.remove(y_pred[i])
                    FN.remove(y_true[j])
                    break

    print("TP: ", TP)
    print("FP: ", FP)
    print("FN: ", FN)

    precision = len(TP) / (len(TP) + len(FP))
    recall = len(TP) / (len(TP) + len(FN))

    F1 = 2 * (precision * recall) / (precision + recall)

    print("precision", precision)
    print("recall", recall)
    print("F1", F1)


class ExtractObjects:
    def __init__(self, frames, cursor_template):
        self.files = frames
        self.sd = None
        self.background = None
        self.object_num = 1
        self.new_object = False
        self.cursor_template = cursor_template
        self.data = defaultdict(dict)
        self.scrolling = False
        # self.data[str(self.object_num)]['bbox'] = []
        self.init()

    def init(self):
        first_frame = cv2.imread(
            os.path.join("assets", "processed", "images", "{}".format(self.files[0]))
        )
        self.sd = ScrollDetector(first_frame)
        self.sd.detect_scroll(first_frame, show=False)
        self.background = first_frame

        if os.path.exists(os.path.join("assets", "processed", "objects")):
            shutil.rmtree(os.path.join("assets", "processed", "objects"))

        os.mkdir(os.path.join("assets", "processed", "objects"))

    def register_object(self, frame, display=False):
        cv2.imwrite(
            os.path.join(
                "assets", "processed", "objects", "{}.png".format(self.object_num)
            ),
            frame,
        )

        print("added object no. ", self.object_num)

        # for box in self.data[str(self.object_num)]['bbox']:
        #     x,y,w,h = box
        #     cv2.rectangle(frame, (x, y),
        #                   (x + w, y + h), (0, 0, 255), 2)

        # re-initialise parmeters to extract new object
        self.object_num += 1
        self.new_object = False
        # self.data[str(self.object_num)]['bbox'] = []

    def add_bbox_coords(self, frame, cursor_template):

        bin_frame = binarize_image(frame, 50)

        # dont put cursor around box
        cursor_pos = detect_cursor(bin_frame, cursor_template)
        cv2.rectangle(bin_frame, cursor_pos[0], cursor_pos[1], 0, -1)

        bin_frame = get_connected_components(bin_frame)

        # display(bin_frame)

        kernel = np.ones((11, 11), np.uint8)
        dilated_mask = cv2.dilate(bin_frame, kernel, iterations=3)

        contours = cv2.findContours(
            dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        contours = contours[0] if len(contours) == 2 else contours[1]

        for cntr in contours:
            x, y, w, h = cv2.boundingRect(cntr)
            self.data[str(self.object_num)]["bbox"].append([x, y, w, h])

    def run(self):

        for f in tqdm(range(0, len(files) - 1)):

            prev_image = cv2.imread(
                os.path.join(
                    "assets", "processed", "images", "{}".format(self.files[f])
                )
            )
            new_image = cv2.imread(
                os.path.join(
                    "assets", "processed", "images", "{}".format(self.files[f + 1])
                )
            )

            # self.sd.detect_scroll(new_image, show=False)
            self.scrolling = False

            if self.sd.scroll_event:
                if not self.scrolling:
                    self.data[str(self.object_num)]["bbox"] = []
                    self.data[str(self.object_num)]["start"] = int(
                        files[f].split(".png")[0]
                    )

                self.scrolling = True
                print(
                    "scrolling activated",
                    "start: ",
                    self.data[str(self.object_num)]["start"],
                )

                self.background = new_image

            else:
                # prev_image = remove_cursor(prev_image, self.cursor_template)
                # new_image = remove_cursor(new_image, self.cursor_template)

                # register when scrolling stops
                if self.scrolling:

                    self.data[str(self.object_num)]["end"] = int(
                        files[f + 1].split(".png")[0]
                    )

                    print("scrolling end: ", self.data[str(self.object_num)]["end"])

                    self.add_bbox_coords(new_image, self.cursor_template)

                    self.register_object(new_image)
                    self.scrolling = False

                else:

                    difference_img = compute_image_difference(new_image, prev_image)
                    # cursor_pos = detect_cursor(difference_img, self.cursor_template)
                    # cv2.rectangle(difference_img, cursor_pos[0], cursor_pos[1], 0, -1)

                    result = get_connected_components(difference_img)

                    # display(result)
                    diff = np.sum(result)
                    # print (diff)

                    if diff == 0 and self.new_object == False:
                        # print ("no new object")
                        continue
                    elif diff == 0 and self.new_object == True:

                        # hull = cv2.convexHull(cntr)
                        # print (hull, "coords")
                        # if w * h > w_max * h_max:
                        #     x_max, y_max, w_max, h_max = x, y, w, h

                        # epsilon = 0.1 * cv2.arcLength(contours[c_num], True)
                        # approx = cv2.approxPolyDP(contours[c_num], epsilon, True)
                        # cv2.drawContours(
                        #     difference, [hull], -1, (0, 0, 255), 1)

                        # cv2.rectangle(difference, (x, y),
                        #               (x + w, y + h), (0, 0, 255), 2)

                        # check whether the object to be recorded is false positive
                        bin_object_img = compute_image_difference(
                            new_image, self.background
                        )
                        # cursor_pos = detect_cursor(bin_object_img, self.cursor_template)
                        # cv2.rectangle(bin_object_img, cursor_pos[0], cursor_pos[1], 0, -1)
                        # display(new_image)
                        # display(self.background)
                        # display(bin_object_img)

                        val = np.sum(get_connected_components(bin_object_img))

                        print(val)

                        if np.sum(val) >= 50000:

                            self.data[str(self.object_num)]["end"] = int(
                                files[f + 1].split(".png")[0]
                            )

                            object_image = compute_image_difference(
                                new_image, self.background, binarize=False
                            )

                            self.add_bbox_coords(object_image, self.cursor_template)
                            self.register_object(object_image)

                            self.background = new_image

                    elif diff > 0 and self.new_object == False:
                        self.new_object = True
                        self.data[str(self.object_num)]["bbox"] = []
                        self.data[str(self.object_num)]["start"] = int(
                            files[f].split(".png")[0]
                        )
                        # print ("detected new object")


if __name__ == "__main__":
    video_name = "eco_short"  # input video name here
    video_type = "mp4"  # generally mp4
    cursor_type = "cursor_eco"  # input cursor type here

    # darker colors need lower threshold
    binarize_threshold = 50
    connected_threshold = 25  # removes video noise

    cursor_template = cv2.imread(
        os.path.join("assets", "cursors", "{}.png".format(cursor_type)), 0
    )

    # Load video (only needed once, comment out once loaded)
    print("video name: {}.{}".format(video_name, video_type))
    load_video(video_name, video_type, undersample=1)

    # Get sampled video frames
    files = [
        i
        for i in os.listdir(os.path.join("assets", "processed", "images"))
        if i.split(".")[-1] == "png"
    ]
    files = sorted(files, key=lambda x: int(x.split(".png")[0]))

    # Extract objects by tracking consecutive frames
    object_extract = ExtractObjects(files, cursor_template)
    object_extract.run()

    # Create json file of objects with start frames
    with open(
        os.path.join("assets", "json", "{}_objects.json".format(video_name)), "w"
    ) as fp:
        json.dump(object_extract.data, fp, indent=1)
