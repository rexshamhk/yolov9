import cv2
import numpy as np
import torch
import os
import platform
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2, increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode


zoom_level = 0
#zoom_center = (0, 0)
zoom_center = None

def rescale_frame(frame, target_width=1920):
    width = frame.shape[1]
    scale_ratio = target_width / width

    # calculate the target height to maintain the aspect ratio
    height = int(frame.shape[0] * scale_ratio)
    dim = (target_width, height)

    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


def handle_mouse_event(event, x, y, flags, param):
    global zoom_level, zoom_center
    if event == cv2.EVENT_MOUSEWHEEL:
        if flags > 0:
            if zoom_level < 6:
                zoom_level += 1
        elif flags < 0:
            if zoom_level > -6:
                zoom_level -= 1
    elif event == cv2.EVENT_LBUTTONDOWN:
        zoom_center = (x, y)

def main():
    global zoom_level, zoom_center
    # Open the video stream
    video_stream = cv2.VideoCapture('rtsp://admin:Insight108!@192.168.140.95:554/Streaming/Channels/201')

    # Check if the video stream is opened successfully
    if not video_stream.isOpened():
        print("Error opening video stream")
        return

    # Set the frame rate to skip frames
    frame_rate = 1  # Adjust this value as per your requirement

    cv2.namedWindow('Video')

    cv2.setMouseCallback('Video', handle_mouse_event)

    # Load model
    #device = select_device(device)
    #model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    #stride, names, pt = model.stride, model.names, model.pt
    #imgsz = check_img_size(imgsz, s=stride)  # check image size


    while True:
        # Read the next frame from the video stream
        ret, frame = video_stream.read()
        if not ret:
            print("Error reading frame")
            continue
         # Handle error

        frame = frame[700:700+1500, 900:900+7260]

        # Perform yolov9 inference on the frame
        # Replace this with your yolov9 inference code


        # Check if the frame was read successfully
        if not ret:
            break

        # Skip frames based on the frame rate
        if video_stream.get(cv2.CAP_PROP_POS_FRAMES) % frame_rate != 0:
            continue

        # Rescale the frame
        frame = rescale_frame(frame)

        if zoom_center is None:
            zoom_center = (frame.shape[1] // 2, frame.shape[0] // 2)

        

        # Zoom in the frame
        if zoom_level != 0:
            h, w = frame.shape[:2]
            x, y = zoom_center
            scale = 1.2 ** zoom_level
            frame_resized = cv2.resize(frame, None, fx=scale, fy=scale)
            
            y_min = max(0, int(y*scale-h/2))
            y_max = min(int(y*scale+h/2), frame_resized.shape[0])
            x_min = max(0, int(x*scale-w/2))
            x_max = min(int(x*scale+w/2), frame_resized.shape[1])
            
            frame_zoomed = frame_resized[y_min:y_max, x_min:x_max]
        
            if zoom_level < 0:
                border_size_x = (w - frame_zoomed.shape[1]) // 2
                border_size_y = (h - frame_zoomed.shape[0]) // 2
                frame = cv2.copyMakeBorder(frame_zoomed, border_size_y, border_size_y, border_size_x, border_size_x, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            else:
                frame = frame_zoomed

        # Show the resulting image on screen
        cv2.imshow('Video', frame)

        # Allow user to use mouse scroll to zoom in the video result
        key = cv2.waitKey(1)
        if key == 27:  # Press 'Esc' key to exit
            break

    # Release the video stream and close the window
    video_stream.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()