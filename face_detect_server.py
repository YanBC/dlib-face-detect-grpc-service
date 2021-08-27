import grpc
from concurrent import futures
import time
from multiprocessing import Process
import os

from protos import face_detect_pb2
from protos import face_detect_pb2_grpc
from configs import face_detect as config
from utils.data_transform import (
    npArray2pbBbox,
    pbImage2npArray
)
from utils.dlib_face_detect import CnnResizedDetect


class FaceDetectServicer(face_detect_pb2_grpc.FaceDetectServicer):
    def __init__(self, gpu_id):
        super().__init__()
        self.det = CnnResizedDetect(gpu_id)

    def GetBboxes(self, request, context):
        image = pbImage2npArray(request.image)
        bbox_list = self.det(image)
        ret = face_detect_pb2.FaceDetectOuput(num=len(bbox_list))
        for bbox in bbox_list:
            ret.bboxes.append(npArray2pbBbox(bbox))
        return ret


def serve_FaceDetectServicer(gpu_id, num=1):
    MAX_MESSAGE_LENGTH = config.MAX_MESSAGE_LENGTH

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=num),
        options=[
            ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
            ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
        ])

    servicer = FaceDetectServicer(gpu_id)
    face_detect_pb2_grpc.add_FaceDetectServicer_to_server(
        servicer, server)

    server.add_insecure_port(f"[::]:{config.PORT}")
    server.start()
    try:
        print("Face Detection is running")
        while True:
            time.sleep(60 * 60 * 24)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    print(os.getpid())

    serve_FaceDetectServicer(0, 1)
