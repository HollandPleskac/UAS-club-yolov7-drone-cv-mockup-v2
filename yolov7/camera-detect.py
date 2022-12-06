# code from https://www.youtube.com/watch?v=XzUMigbYRUI
import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


def displayArrows(xmin, ymin, xmax, ymax, center_x, center_y, screen_center_x, screen_center_y, width, height, show_img):
    # Variables for Text
    text = "Need to move up"
    coordinates = (100, height-100)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 4
    color = (255, 0, 0)
    thickness = 10

    if (xmax-xmin) * (ymax-ymin) > 200*200:
        show_img = cv2.putText(show_img, text, coordinates,
                               font, fontScale, color, thickness, cv2.LINE_AA)
    else:
        # Display Arrows
        if (xmin < screen_center_x - 100):
            cv2.arrowedLine(show_img, (width-150, screen_center_y),
                            (width-50, screen_center_y), (0, 255, 0), 30, tipLength=0.5)
        elif (xmax > screen_center_x+100):
            cv2.arrowedLine(show_img, (150, screen_center_y),
                            (50, screen_center_y), (0, 255, 0), 30, tipLength=0.5)
        else:
            cv2.circle(show_img, (center_x, center_y), 15, (0, 255, 0), -1)

        if (ymin < screen_center_y - 100):
            cv2.arrowedLine(show_img, (screen_center_x, height-150),
                            (screen_center_x, height-50), (0, 255, 0), 30, tipLength=0.5)
        elif (ymax > screen_center_y+100):
            cv2.arrowedLine(show_img, (screen_center_x, 150),
                            (screen_center_x, 50), (0, 255, 0), 30, tipLength=0.5)
        else:
            cv2.circle(show_img, (center_x, center_y), 15, (0, 255, 0), -1)


def detect(source, weights, device, img_size, iou_thres, conf_thres):

    webcam = source.isnumeric()

    # Initialize
    set_logging()
    device = select_device(device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load Model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(img_size, s=stride)  # check img_size

    if half:
        model.half()  # to FP16

    # Set Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(
            next(model.parameters())))  # run once

    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres)
        t3 = time_synchronized()

        # Screen dimensions
        height = img.shape[2]
        width = img.shape[3]

        # Screen Center
        screen_center_x = width / 2
        screen_center_y = height / 2
        screen_center_x, screen_center_y = int(
            screen_center_x), int(screen_center_y)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(
                ), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path

            # normalization gain whwh
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    # add to string
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    label = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, im0, label=label,
                                 color=colors[int(cls)], line_thickness=1)
                    c1, c2 = (int(xyxy[0]), int(xyxy[1])
                              ), (int(xyxy[2]), int(xyxy[3]))
                    print(c1, c2)

                    xmin = c1[0]
                    ymin = c1[1]
                    xmax = c2[0]
                    ymax = c2[1]
                    center_x = (xmin + xmax)/2
                    center_y = (ymin + ymax)/2
                    center_x, center_y = int(center_x), int(center_y)

                    displayArrows(xmin, ymin, xmax, ymax, center_x, center_y,
                                  screen_center_x, screen_center_y, width, height, im0)

            # Print time (inference + NMS)
            #print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

        # ---------  Display target box rectangle -------------

        cv2.rectangle(im0, (screen_center_x-100, screen_center_y-100),
                      (screen_center_x+100, screen_center_y+100), (0, 255, 0), 3)

        cv2.imshow(str(p), im0)
#         cv2.waitKey(1)  # 1 millisecond

    print(f'Done. ({time.perf_counter() - t0:.3f}s)')


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(device)

    with torch.no_grad():
        detect("1", "runs/train/yolov7-custom3/weights/best.pt",
               device, img_size=640, iou_thres=0.45, conf_thres=0.2)
