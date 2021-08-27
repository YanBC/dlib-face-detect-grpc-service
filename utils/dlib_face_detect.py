import dlib
import cv2 as cv
import numpy as np
from typing import Tuple
import os


class HogDetect:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()

    def __call__(self, image) -> np.ndarray:
        dlib_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        dets = self.detector(dlib_image, 1)

        ret = []
        for det in dets:
            left = det.left()
            top = det.top()
            right = det.right()
            bottom = det.bottom()
            ret.append([left, top, right, bottom])

        return np.asarray(ret)


class CnnResizedDetect:
    def __init__(self, gpu_id,
            model_path="models/mmod_human_face_detector.dat"):
        old_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_id}"
        self.detector = dlib.cnn_face_detection_model_v1(model_path)
        if old_devices is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = old_devices
        else:
            os.environ.pop("CUDA_VISIBLE_DEVICES")

    def inference(self, dlib_image) -> np.ndarray:
        dets = self.detector(dlib_image, 1)

        ret = []
        for det in dets:
            left = det.rect.left()
            top = det.rect.top()
            right = det.rect.right()
            bottom = det.rect.bottom()
            ret.append([left, top, right, bottom])

        return np.asarray(ret)

    @classmethod
    def preprocess(cls, image, target_size=1024) -> Tuple[np.ndarray, float]:
        image_h, image_w = image.shape[0:2]
        if image_h > image_w:
            scale = target_size / image_h
        else:
            scale = target_size / image_w
        target_h = int(image_h * scale)
        target_w = int(image_w * scale)

        dlib_image = cv.resize(image, (target_w, target_h))
        dlib_image = cv.cvtColor(dlib_image, cv.COLOR_BGR2RGB)
        return dlib_image, scale

    @classmethod
    def postprocess(cls, scale, bboxes) -> np.ndarray:
        ret_bboxes = bboxes / scale
        return ret_bboxes.astype(np.int)

    def __call__(self, image) -> np.ndarray:
        dlib_image, scale = self.preprocess(image)
        bboxes = self.inference(dlib_image)
        ret_bboxes = self.postprocess(scale, bboxes)
        return ret_bboxes


if __name__ == '__main__':
    from visualize_image import draw_bbox
    import sys
    image_path = sys.argv[1]
    image = cv.imread(image_path)

    # det_net = HogDetect()
    CnnDetectModelPath = 'models/mmod_human_face_detector.dat'
    det_net = CnnResizedDetect(0, CnnDetectModelPath)

    bboxes = det_net(image)
    canvas = image.copy()
    for bbox in bboxes:
        left, top, right, bottom = bbox
        width = right - left + 1
        height = bottom - top + 1
        canvas = draw_bbox(canvas, [left, top, right, bottom])

    cv.imwrite("result.png", canvas)
