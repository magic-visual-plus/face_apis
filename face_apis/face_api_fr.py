
from .face_api import FaceApi
import face_recognition
from typing import List, Tuple
import scipy.spatial as spatial
import numpy as np


class FaceApiFaceRecognition(FaceApi):
    def __init__(self, model_folder):
        super().__init__(model_folder)
        pass

    def build_index(self, ids: List[str], images: List[np.ndarray]):
        self.ids = []
        face_encodings = []
        for i, image in enumerate(images):
            print(ids[i])
            face_locations = face_recognition.face_locations(image, model='cnn')
            if len(face_locations) == 0:
                print(f'No face found in image: {ids[i]}')
                continue

            for face_location in face_locations:
                face_encoding = face_recognition.face_encodings(image, [face_location])[0]
                face_encodings.append(face_encoding)
                self.ids.append(ids[i])
                pass
            pass

        face_encodings = np.asarray(face_encodings)
        
        self.tree = spatial.KDTree(face_encodings)


    def detect_and_search(self, image: np.ndarray, threshold: float = 0.6) -> List[Tuple[Tuple[int], str, float]]:
        face_locations = face_recognition.face_locations(image, model='cnn')
        results = []
        if len(face_locations) == 0:
            return []
        else:
            for face_location in face_locations:
                print(face_location)
                face_encoding = face_recognition.face_encodings(image, [face_location])[0]
                distances, indices = self.tree.query([face_encoding], k=1)
                distance = distances[0]
                index = indices[0]
                if distance < threshold:
                    results.append(
                        (face_location, self.ids[index], distance)
                    )
                else:
                    results.append(
                        (face_location, None, None)
                    )
                pass
            return results
        pass