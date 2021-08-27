import cv2 as cv


def draw_bbox(image, bbox):
    canvas = image.copy()
    left, top, right, bottom = bbox
    cv.rectangle(canvas, (left, top), (right, bottom), (0, 255, 0), 3)
    return canvas


def draw_text(image, bbox, text):
    canvas = image.copy()
    left, top, right, bottom = bbox
    
    labelSize, baseLine = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 1, 3)
    
    text_bbox_left = left
    text_bbox_top = top + 10
    text_bbox_right = text_bbox_left + labelSize[0]
    text_bbox_bottom = text_bbox_top + labelSize[1]
    cv.rectangle(
        canvas,
        (text_bbox_left, text_bbox_top),
        (text_bbox_right, text_bbox_bottom),
        (0, 0, 255),
        cv.FILLED)
    
    cv.putText(
        canvas,
        text,
        (text_bbox_left, text_bbox_bottom),
        cv.FONT_HERSHEY_SIMPLEX,
        1,
        (0,0,0),
        3)
    return canvas
