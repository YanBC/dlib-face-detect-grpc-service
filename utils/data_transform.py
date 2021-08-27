from builtins import float
import numpy as np
import cv2 as cv
from protos import common_pb2


def pbBbox2npArray(pbBbox: common_pb2.Bbox) -> np.ndarray:
    box = np.asarray([pbBbox.left, pbBbox.top, pbBbox.right, pbBbox.bottom])
    return box


def npArray2pbBbox(npArray: np.ndarray) -> common_pb2.Bbox:
    assert npArray.shape == (4,)
    tmp_arr = npArray.astype(float)
    left, top, right, bottom = tmp_arr
    return common_pb2.Bbox(
        left=left, top=top, right=right, bottom=bottom)


def pbImage2npArray(pbImage: common_pb2.Image) -> np.ndarray:
    b_str = pbImage.filebytes
    nparr = np.fromstring(b_str, np.uint8)
    image = cv.imdecode(nparr, cv.IMREAD_UNCHANGED)
    return image


def file2pbImage(image_path: str) -> common_pb2.Image:
    with open(image_path, 'br') as f:
        b_str = f.read()
    return common_pb2.Image(filebytes=b_str)
