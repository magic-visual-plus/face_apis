import unittest
from face_apis import face_api_fr
import cv2
import os
import time


class TestFaceRecognition(unittest.TestCase):
    def test_face_recognition(self):
        face_api = face_api_fr.FaceApiFaceRecognition('')

        current_path = os.path.dirname(os.path.realpath(__file__))
        base_path = os.path.join(current_path, 'data', 'base')

        filenames = os.listdir(base_path)
        filenames = [filename for filename in filenames if filename.endswith('.jpg')]
        ids = [filename.split('.')[0] for filename in filenames]
        images = []
        for filename in filenames:
            image = face_api_fr.face_recognition.load_image_file(os.path.join(base_path, filename))
            max_size = 800
            if image.shape[0] > max_size or image.shape[1] > max_size:
                scale = max_size / max(image.shape[0], image.shape[1])
                image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
                pass

            images.append(image)
            pass

        face_api.build_index(ids, images)

        query_path = os.path.join(current_path, 'data', 'query')
        filenames = os.listdir(query_path)
        filenames = [filename for filename in filenames if filename.endswith('.jpg')]
        for filename in filenames:
            image = face_api_fr.face_recognition.load_image_file(os.path.join(query_path, filename))
            max_size = 800
            if image.shape[0] > max_size or image.shape[1] > max_size:
                scale = max_size / max(image.shape[0], image.shape[1])
                image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
                pass
            basename = filename.split('.')[0]
            start = time.time()
            results = face_api.detect_and_search(image)
            print(f'{filename}: {time.time() - start}')
            self.assertEqual(len(results), 1)
            self.assertEqual(basename, results[0][1])
            pass
        pass
    pass


if __name__ == '__main__':
    unittest.main()
    pass