import face_recognition as fr
import numpy as np


class Detector:
    def __init__(self, model='hog'):
        self.model = model

    def get_box(self, img):
        faces = fr.face_locations(img, model=self.model)
        if not faces:
            raise Exception('No face detected')
        return faces[0]

    def get_eyes(self, img):
        face = self.get_box(img)
        landmark = fr.face_landmarks(
            face_image=img,
            face_locations=[face],
            model='small'
        )[0]
        left_eye = np.subtract(
            np.mean(landmark['left_eye'], 0),
            (face[-1], face[0])
        )
        right_eye = np.subtract(
            np.mean(landmark['right_eye'], 0),
            (face[-1], face[0])
        )
        return left_eye, right_eye
