import grpc
import cv2 as cv
import numpy as np
from typing import Tuple, List

from protos import common_pb2
from protos import face_detect_pb2
from protos import face_detect_pb2_grpc
from configs import face_detect as config
from utils.data_transform import (
    file2pbImage,
    pbBbox2npArray
)
from utils.visualize_image import draw_bbox


class FaceDetect:
    def __init__(self):
        MAX_MESSAGE_LENGTH = config.MAX_MESSAGE_LENGTH
        channel = grpc.insecure_channel(
            f"localhost:{config.PORT}",
            options=[
                ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
                ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
            ])
        self.stub = face_detect_pb2_grpc.FaceDetectStub(channel)

    def run(self, image_path) -> List[np.ndarray]:
        pb_image = file2pbImage(image_path)
        request = face_detect_pb2.FaceDetectInput(image=pb_image)
        response = self.stub.GetBboxes(request)

        bboxes = []
        for bbox in response.bboxes:
            bboxes.append(pbBbox2npArray(bbox))

        return bboxes


def simple_run(image_path):
    net = FaceDetect()
    bboxes = net.run(image_path)

    canvas = cv.imread(image_path)
    for bbox in bboxes:
        bbox = bbox.astype(np.int)
        canvas = draw_bbox(canvas, bbox)
    return canvas


if __name__ == '__main__':
    import os
    import sys
    image_path = sys.argv[1]
    print(f"pid: {os.getpid()}")

    canvas = simple_run(image_path)
    cv.imwrite('result.png', canvas)
