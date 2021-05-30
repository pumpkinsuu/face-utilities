import mtcnn
import numpy as np


class Detector:
    def __init__(self):
        self.detector = mtcnn.MTCNN()

    def get_face(self, img):
        faces = self.detector.detect_faces(img)
        if not faces:
            raise Exception('No face detected')
        return faces[0]

    def get_box(self, img):
        face = self.get_face(img)
        x, y, width, height = face['box']

        left = x * (x > 0)
        top = y * (y > 0)
        right = x + width
        bottom = y + height

        return top, right, bottom, left

    def get_eyes(self, img):
        face = self.get_face(img)
        x, y, width, height = face['box']

        left_eye = np.subtract(face["keypoints"]["left_eye"], (x, y))
        right_eye = np.subtract(face["keypoints"]["right_eye"], (x, y))

        return left_eye, right_eye
