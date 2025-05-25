# from mediapipe import solutions
# from mediapipe.framework.formats import landmark_pb2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import torch

def load_mediapipe(model_path):
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                        output_face_blendshapes=True,
                                        output_facial_transformation_matrixes=False,
                                        num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)
    return detector

def get_landmark(detector, image):
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    detection_result = detector.detect(image)
    if detection_result.face_landmarks:
        landmarks = torch.tensor([[lm.x, lm.y, lm.z] for lm in detection_result.face_landmarks[0]], dtype=torch.float32)
    else:
        landmarks = torch.zeros((478, 3), dtype=torch.float32)
    return landmarks