import pydantic
from typing import List, Tuple
import numpy as np


class FaceApi(object):
    def __init__(self, model_folder):
        pass

    def build_index(self, ids: List[str], images: List[np.ndarray]):
        # image should RGB array
        pass

    def detect_and_search(self, image: np.ndarray, threshold: float = 0.6) -> List[Tuple[Tuple[int], str, float]]:
        # image should RGB array

        # returns: list of (face_location, id, distance), face_location is a tuple of (x, y, w, h)
        pass