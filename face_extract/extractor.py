import numpy as np
from PIL import Image


def distance(a, b):
    return np.sqrt(np.sum((a - b)**2))


class Extractor:
    def __init__(self, model='hog'):
        if model == 'hog' or model == 'cnn':
            from face_extract.fr_detect import Detector
            self.detector = Detector(model)
        else:
            from face_extract.mtcnn_detect import Detector
            self.detector = Detector()

    def extract(self, img, align=True):
        if align:
            left_eye, right_eye = self.detector.get_eyes(img)
            left_eye_x, left_eye_y = left_eye
            right_eye_x, right_eye_y = right_eye

            if left_eye_y > right_eye_y:
                point_3rd = np.array([right_eye_x, left_eye_y])
                direction = -1
            else:
                point_3rd = np.array([left_eye_x, right_eye_y])
                direction = 1

            a = distance(left_eye, point_3rd)
            b = distance(right_eye, point_3rd)
            c = distance(right_eye, left_eye)

            if a != 0 and b != 0:
                cos_a = (b * b + c * c - a * a) / (2 * b * c)
                angle = np.arccos(cos_a)
                angle = (angle * 180) / np.pi

                if direction == -1:
                    angle = 90 - angle

                img = Image.fromarray(img)
                img = np.array(img.rotate(direction * angle))

        top, right, bottom, left = self.detector.get_box(img)
        return img[top:bottom, left:right]
