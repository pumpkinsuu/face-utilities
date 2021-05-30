import numpy as np
import mtcnn
from PIL import Image


def distance(a, b):
    return np.sqrt(np.sum((a - b)**2))


class Extractor:
    def __init__(self):
        self.detector = mtcnn.MTCNN()

    def _get_face(self, img):
        faces = self.detector.detect_faces(img)
        if not faces:
            raise Exception('No face detected')
        return faces[0]

    def get_face(self, img):
        faces = self.detector.detect_faces(img)
        if not faces:
            raise Exception('No face detected')
        return faces[0]

    def _crop(self, img):
        face = self.get_face(img)
        x, y, width, height = face['box']

        left = x * (x > 0)
        top = y * (y > 0)
        right = x + width
        bottom = y + height

        return img[top:bottom, left:right]

    def _align(self, img):
        face = self.get_face(img)
        left_eye = np.array(face["keypoints"]["left_eye"])
        right_eye = np.array(face["keypoints"]["right_eye"])
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

            _img = Image.fromarray(img)
            _img = np.array(_img.rotate(direction * angle, expand=True))
            return _img
        return img

    def extract(self, img, align=True):
        _img = np.array(img)
        if align:
            _img = self._align(_img)
        return self._crop(_img)
