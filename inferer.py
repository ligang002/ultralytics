import math

import cv2
import numpy as np
import torch

from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils.ops import non_max_suppression, scale_boxes
from ultralytics import YOLO
from ultralytics.utils.torch_utils import select_device
from ultralytics.utils import ops


def make_divisible(x, divisor):
    # Returns nearest x divisible by divisor
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor


def check_img_size(imgsz, s=32, floor=0):
    # Verify image size is a multiple of stride s in each dimension
    if isinstance(imgsz, int):  # integer i.e. img_size=640
        new_size = max(make_divisible(imgsz, int(s)), floor)
    else:  # list i.e. img_size=[640, 480]
        imgsz = list(imgsz)  # convert to list if tuple
        new_size = [max(make_divisible(x, int(s)), floor) for x in imgsz]
    if new_size != imgsz:
        LOGGER.warning(f'WARNING ⚠️ --img-size {imgsz} must be multiple of max stride {s}, updating to {new_size}')
    return new_size


class YOLOV8:
    '''
    weigth:model path
    '''

    def __init__(self, weights, cuda, verbose) -> None:
        # super().__init__(weight)
        self.imgsz = 640, 640
        self.cuda = cuda
        self.model = AutoBackend(weights, device=select_device(cuda, verbose=verbose), verbose=verbose)
        print(self.model)
        names = self.model.names
        print(names)
        self.names = names
        self.half = False
        self.conf = 0.25
        self.iou = 0.7
        self.stride = 32
        self.model.eval()

    def infer(self, img_src, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic_nms=False, max_det=100):
        """
        yolov5 image infer \n
        :param img_src: source image numpy format
        :param conf_thres: Confidence Threshold
        :param iou_thres:   IOU Threshold
        :param classes: classes
        :return results: detection results: list [['person', 0.95, [3393, 811, 3836, 1417]], ...] 左上角、右下角
        """

        img = self.precess_image(img_src, self.imgsz, self.stride, self.half, self.cuda)
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

        preds = self.model(img)

        det = ops.non_max_suppression(preds, self.conf, self.iou)

        # 格式转换
        lst_result = []
        if len(det):
            for i, pred in enumerate(det):
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], img_src.shape)
                print(pred)
                results = pred.cpu().detach().numpy()
                # 格式转换
                for detection in results:
                    index = detection[5]  # 获取类别索引
                    cls_name = self.names.get(index, 'unknown')  # 根据索引获取类别名称，未知类别则设置为'unknown'
                    new_detection = [cls_name, detection[4], [detection[0], detection[1], detection[2], detection[3]]]  # 构建新的检测结果项
                    lst_result.append(new_detection)

        return lst_result

    def letterbox(self, im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True,
                  stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im, ratio, (dw, dh)

    def precess_image(self, img_src, img_size, stride, half, device):
        '''Process image before image inference.'''
        # Padded resize
        img = self.letterbox(img_src, img_size, stride=stride)[0]
        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)

        img = img.half() if half else img.float()  # uint8 to fp16/32
        img = img / 255  # 0 - 255 to 0.0 - 1.0
        return img


if __name__ == '__main__':
    weights = 'best.pt'
    cuda = 'cuda:0'
    verbose = False

    yolov8 = YOLOV8(weights, cuda, verbose)

    img_src = cv2.imread(r'01-90_85-274&482_457&544-456&545_270&533_261&470_447&482-0_0_16_24_31_28_31-146-27.jpg')
    lst_result = yolov8.infer(img_src)
    print(lst_result)
