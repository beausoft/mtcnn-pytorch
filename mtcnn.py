import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import math
import numpy as np
import os
from PIL import Image


__all__ = ['MTCNN']


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        # without this pretrained model isn't working
        x = x.transpose(3, 2).contiguous()
        # 将[batch_size, c, h, w]变换为[batch_size, c*h*w].
        return x.view(x.size(0), -1)


class PNet(nn.Module):

    def __init__(self):
        super(PNet, self).__init__()
        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 10, 3, 1)),
            ('prelu1', nn.PReLU(10)),
            ('pool1', nn.MaxPool2d(2, 2, ceil_mode=True)),

            ('conv2', nn.Conv2d(10, 16, 3, 1)),
            ('prelu2', nn.PReLU(16)),

            ('conv3', nn.Conv2d(16, 32, 3, 1)),
            ('prelu3', nn.PReLU(32))
        ]))

        self.conv4_1 = nn.Conv2d(32, 2, 1, 1)
        self.conv4_2 = nn.Conv2d(32, 4, 1, 1)

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [batch_size, 3, h, w].
        Returns:
            b: a float tensor with shape [batch_size, 4, h', w'].
            a: a float tensor with shape [batch_size, 2, h', w'].
        """
        y = self.features(x)
        probs = F.softmax(self.conv4_1(y), dim=1)
        offsets = self.conv4_2(y)
        return offsets, probs


class RNet(nn.Module):

    def __init__(self):
        super(RNet, self).__init__()
        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 28, 3, 1)),
            ('prelu1', nn.PReLU(28)),
            ('pool1', nn.MaxPool2d(3, 2, ceil_mode=True)),

            ('conv2', nn.Conv2d(28, 48, 3, 1)),
            ('prelu2', nn.PReLU(48)),
            ('pool2', nn.MaxPool2d(3, 2, ceil_mode=True)),

            ('conv3', nn.Conv2d(48, 64, 2, 1)),
            ('prelu3', nn.PReLU(64)),

            ('flatten', Flatten()),
            ('conv4', nn.Linear(576, 128)),
            ('prelu4', nn.PReLU(128))
        ]))

        self.conv5_1 = nn.Linear(128, 2)
        self.conv5_2 = nn.Linear(128, 4)

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [batch_size, 3, h, w].
        Returns:
            b: a float tensor with shape [batch_size, 4].
            a: a float tensor with shape [batch_size, 2].
        """
        y = self.features(x)
        probs = F.softmax(self.conv5_1(y), dim=1)
        offsets = self.conv5_2(y)
        return offsets, probs


class ONet(nn.Module):

    def __init__(self):
        super(ONet, self).__init__()
        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 32, 3, 1)),
            ('prelu1', nn.PReLU(32)),
            ('pool1', nn.MaxPool2d(3, 2, ceil_mode=True)),

            ('conv2', nn.Conv2d(32, 64, 3, 1)),
            ('prelu2', nn.PReLU(64)),
            ('pool2', nn.MaxPool2d(3, 2, ceil_mode=True)),

            ('conv3', nn.Conv2d(64, 64, 3, 1)),
            ('prelu3', nn.PReLU(64)),
            ('pool3', nn.MaxPool2d(2, 2, ceil_mode=True)),

            ('conv4', nn.Conv2d(64, 128, 2, 1)),
            ('prelu4', nn.PReLU(128)),

            ('flatten', Flatten()),
            ('conv5', nn.Linear(1152, 256)),
            ('drop5', nn.Dropout(0.25)),
            ('prelu5', nn.PReLU(256)),
        ]))

        self.conv6_1 = nn.Linear(256, 2)
        self.conv6_2 = nn.Linear(256, 4)
        self.conv6_3 = nn.Linear(256, 10)

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [batch_size, 3, h, w].
        Returns:
            c: a float tensor with shape [batch_size, 10].
            b: a float tensor with shape [batch_size, 4].
            a: a float tensor with shape [batch_size, 2].
        """
        y = self.features(x)
        probs = F.softmax(self.conv6_1(y), dim=1)
        offsets = self.conv6_2(y)
        landmarks = self.conv6_3(y)
        return landmarks, offsets, probs


def _preprocess(img):
    """
    Preprocessing step before feeding the network.
    Arguments:
        img: a float numpy array of shape [h, w, c].
    Returns:
        a float numpy array of shape [1, c, h, w].
    """
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, 0)
    img = (img - 127.5) * 0.0078125
    return img


def _generate_bboxes(probs, offsets, scale, threshold):
    """
    Generate bounding boxes at places where there is probably a face.
    Arguments:
        probs: a float numpy array of shape [n, m].
        offsets: a float numpy array of shape [1, 4, n, m].
        scale: a float number, width and height of the image were scaled by this number.
        threshold: a float number.
    Returns:
        a float numpy array of shape [n_boxes, 9]
    """
    # applying P-Net is equivalent, in some sense, to moving 12x12 window with stride 2
    stride = 2
    cell_size = 12
    # indices of boxes where there is probably a face
    inds = np.where(probs > threshold)
    if inds[0].size == 0:
        return np.array([])
    # transformations of bounding boxes
    tx1, ty1, tx2, ty2 = [offsets[0, i, inds[0], inds[1]] for i in range(4)]
    # they are defined as:
    # w = x2 - x1 + 1
    # h = y2 - y1 + 1
    # x1_true = x1 + tx1*w
    # x2_true = x2 + tx2*w
    # y1_true = y1 + ty1*h
    # y2_true = y2 + ty2*h
    offsets = np.array([tx1, ty1, tx2, ty2])
    score = probs[inds[0], inds[1]]
    # P-Net is applied to scaled images so we need to rescale bounding boxes back
    bounding_boxes = np.vstack([
        np.round((stride*inds[1] + 1.0)/scale),
        np.round((stride*inds[0] + 1.0)/scale),
        np.round((stride*inds[1] + 1.0 + cell_size)/scale),
        np.round((stride*inds[0] + 1.0 + cell_size)/scale),
        score, offsets
    ])
    # why one is added?
    return bounding_boxes.T


def _nms(boxes, overlap_threshold=0.5, mode='union'):
    """
    Non-maximum suppression.
    Arguments:
        boxes: a float numpy array of shape [n, 5], where each row is (xmin, ymin, xmax, ymax, score).
        overlap_threshold: a float number.
        mode: 'union' or 'min'.
    Returns:
        list with indices of the selected boxes
    """
    # if there are no boxes, return the empty list
    if len(boxes) == 0:
        return []
    # list of picked indices
    pick = []
    # grab the coordinates of the bounding boxes
    x1, y1, x2, y2, score = [boxes[:, i] for i in range(5)]
    area = (x2 - x1 + 1.0) * (y2 - y1 + 1.0)
    ids = np.argsort(score)  # in increasing order
    while len(ids) > 0:
        # grab index of the largest value
        last = len(ids) - 1
        i = ids[last]
        pick.append(i)

        # compute intersections of the box with the largest score with the rest of boxes

        # left top corner of intersection boxes
        ix1 = np.maximum(x1[i], x1[ids[:last]])
        iy1 = np.maximum(y1[i], y1[ids[:last]])

        # right bottom corner of intersection boxes
        ix2 = np.minimum(x2[i], x2[ids[:last]])
        iy2 = np.minimum(y2[i], y2[ids[:last]])

        # width and height of intersection boxes
        w = np.maximum(0.0, ix2 - ix1 + 1.0)
        h = np.maximum(0.0, iy2 - iy1 + 1.0)

        # intersections' areas
        inter = w * h
        if mode == 'min':
            overlap = inter/np.minimum(area[i], area[ids[:last]])
        elif mode == 'union':
            # intersection over union (IoU)
            overlap = inter/(area[i] + area[ids[:last]] - inter)

        # delete all boxes where overlap is too big
        ids = np.delete(ids, np.concatenate([[last], np.where(overlap > overlap_threshold)[0]]))
    return pick


def _convert_to_square(bboxes):
    """
    Convert bounding boxes to a square form.
    Arguments:
        bboxes: a float numpy array of shape [n, 5].
    Returns:
        a float numpy array of shape [n, 5], squared bounding boxes.
    """
    square_bboxes = np.zeros_like(bboxes)
    x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
    h = y2 - y1 + 1.0
    w = x2 - x1 + 1.0
    max_side = np.maximum(h, w)
    square_bboxes[:, 0] = x1 + w*0.5 - max_side*0.5
    square_bboxes[:, 1] = y1 + h*0.5 - max_side*0.5
    square_bboxes[:, 2] = square_bboxes[:, 0] + max_side - 1.0
    square_bboxes[:, 3] = square_bboxes[:, 1] + max_side - 1.0
    return square_bboxes


def _calibrate_box(bboxes, offsets):
    """
    Transform bounding boxes to be more like true bounding boxes. 'offsets' is one of the outputs of the nets.
    Arguments:
        bboxes: a float numpy array of shape [n, 5].
        offsets: a float numpy array of shape [n, 4].
    Returns:
        a float numpy array of shape [n, 5].
    """
    x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
    w = x2 - x1 + 1.0
    h = y2 - y1 + 1.0
    w = np.expand_dims(w, 1)
    h = np.expand_dims(h, 1)

    # this is what happening here:
    # tx1, ty1, tx2, ty2 = [offsets[:, i] for i in range(4)]
    # x1_true = x1 + tx1*w
    # y1_true = y1 + ty1*h
    # x2_true = x2 + tx2*w
    # y2_true = y2 + ty2*h
    # below is just more compact form of this

    # are offsets always such that
    # x1 < x2 and y1 < y2 ?

    translation = np.hstack([w, h, w, h])*offsets
    bboxes[:, 0:4] = bboxes[:, 0:4] + translation
    return bboxes


def _get_image_boxes(bounding_boxes, img, size=24):
    """
    Cut out boxes from the image.
    Arguments:
        bounding_boxes: a float numpy array of shape [n, 5].
        img: an instance of PIL.Image.
        size: an integer, size of cutouts.
    Returns:
        a float numpy array of shape [n, 3, size, size].
    """
    num_boxes = len(bounding_boxes)
    width, height = img.size
    # 对回归框进行裁剪防止超出图像范围
    [dy, edy, dx, edx, y, ey, x, ex, w, h] = _correct_bboxes(bounding_boxes, width, height)
    img_boxes = np.zeros((num_boxes, 3, size, size), dtype=np.float32)

    for i in range(num_boxes):
        img_box = np.zeros((h[i], w[i], 3), dtype=np.uint8)

        img_array = np.asarray(img, dtype=np.uint8)
        img_box[dy[i]:(edy[i] + 1), dx[i]:(edx[i] + 1), :] = img_array[y[i]:(ey[i] + 1), x[i]:(ex[i] + 1), :]  # 对图像进行裁剪

        # resize
        img_box = Image.fromarray(img_box)  # 把numpy转换回PIL.Image
        img_box = img_box.resize((size, size), Image.BILINEAR)  # 对图像进行缩放
        img_box = np.asarray(img_box, dtype=np.float32)

        img_boxes[i, :, :, :] = _preprocess(img_box)

    return img_boxes


def _correct_bboxes(bboxes, width, height):
    """
    Crop boxes that are too big and get coordinates with respect to cutouts.
    Arguments:
        bboxes: a float numpy array of shape [n, 5],
            where each row is (xmin, ymin, xmax, ymax, score).
        width: a float number.
        height: a float number.
    Returns:
        dy, dx, edy, edx: a int numpy arrays of shape [n], coordinates of the boxes with respect to the cutouts.
        y, x, ey, ex: a int numpy arrays of shape [n], corrected ymin, xmin, ymax, xmax.
        h, w: a int numpy arrays of shape [n], just heights and widths of boxes.
        in the following order: [dy, edy, dx, edx, y, ey, x, ex, w, h].
    """

    x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
    w, h = x2 - x1 + 1.0,  y2 - y1 + 1.0
    num_boxes = bboxes.shape[0]

    # 'e' stands for end
    # (x, y) -> (ex, ey)
    x, y, ex, ey = x1, y1, x2, y2

    # we need to cut out a box from the image.
    # (x, y, ex, ey) are corrected coordinates of the box
    # in the image.
    # (dx, dy, edx, edy) are coordinates of the box in the cutout
    # from the image.
    dx, dy = np.zeros((num_boxes,)), np.zeros((num_boxes,))
    edx, edy = w.copy() - 1.0, h.copy() - 1.0

    # if box's bottom right corner is too far right
    ind = np.where(ex > width - 1.0)[0]
    edx[ind] = w[ind] + width - 2.0 - ex[ind]
    ex[ind] = width - 1.0

    # if box's bottom right corner is too low
    ind = np.where(ey > height - 1.0)[0]
    edy[ind] = h[ind] + height - 2.0 - ey[ind]
    ey[ind] = height - 1.0

    # if box's top left corner is too far left
    ind = np.where(x < 0.0)[0]
    dx[ind] = 0.0 - x[ind]
    x[ind] = 0.0

    # if box's top left corner is too high
    ind = np.where(y < 0.0)[0]
    dy[ind] = 0.0 - y[ind]
    y[ind] = 0.0

    return_list = [dy, edy, dx, edx, y, ey, x, ex, w, h]
    return_list = [i.astype(np.int32) for i in return_list]

    return return_list


class MTCNN(nn.Module):

    def __init__(self, pth='mtcnn.pth'):
        super(MTCNN, self).__init__()
        self.pnet = PNet()
        self.rnet = RNet()
        self.onet = ONet()
        if os.path.exists(pth) and os.path.isfile(pth):
            state_dict = torch.load(pth, map_location='cpu')
            self.load_state_dict(state_dict)
        self.is_cuda = False

    def __run_first_stage(self, image, scale, threshold):
        '''
        运行P-Net，生成边界框，并执行NMS。
        Arguments:
            image: an instance of PIL.Image.
            net: an instance of pytorch's nn.Module, P-Net.
            scale: a float number, scale width and height of the image by this number.
            threshold: a float number, threshold on the probability of a face when generating bounding boxes from predictions of the net.
        Returns:
            a float numpy array of shape [n_boxes, 9], bounding boxes with scores and offsets (4 + 1 + 4).
        '''
        # 缩放图像并将其转换为float数组
        width, height = image.size
        sw, sh = math.ceil(width*scale), math.ceil(height*scale)
        img = image.resize((sw, sh), Image.BILINEAR)  # 根据比例缩放图像
        # 对图像进行归一化标准化并转换为tensor
        img = torch.tensor(_preprocess(np.asarray(img, dtype=np.float32)))
        if self.is_cuda:
            img = img.cuda()
        offsets, probs = self.pnet(img)  # 调用模型
        # 将模型输出结果转回numpy
        if self.is_cuda:
            offsets = offsets.cpu()
            probs = probs.cpu()
        offsets = offsets.numpy()   # offsets: transformations to true bounding boxes
        probs = probs.numpy()[0, 1, :, :]  # probs: probability of a face at each sliding window
        boxes = _generate_bboxes(probs, offsets, scale, threshold)
        if len(boxes) == 0:
            return None
        keep = _nms(boxes[:, 0:5], overlap_threshold=0.5)
        return boxes[keep]

    def forward(self, image, min_face_size=20.0, thresholds=[0.6, 0.7, 0.8], nms_thresholds=[0.7, 0.7, 0.7]):
        '''
        模型为eval时：
        Arguments:
            image: PIL图像
            min_face_size: 最小人脸 float类型
            thresholds:  三个子网络的阈值 输入元组长度为3
            nms_thresholds  三个子网络对应的NMS阈值 输入元组长度为3
        Returns:
            two float numpy arrays of shapes [n_boxes, 4] and [n_boxes, 10],
            bounding boxes and facial landmarks.
        '''
        if self.training:
            raise NotImplementedError("训练部分暂未实现，现在的权重来自官方转换过来的")
        # BUILD AN IMAGE PYRAMID  构建图像金字塔
        width, height = image.size
        min_length = min(height, width)

        min_detection_size = 12
        factor = 0.707  # sqrt(0.5)
        # 图像的缩放比例
        scales = []
        # 缩放图像，使我们能检测到的最小尺寸等于我们想要检测到的最小脸部尺寸
        m = min_detection_size / min_face_size
        min_length *= m
        factor_count = 0
        while min_length > min_detection_size:
            scales.append(m * factor ** factor_count)
            min_length *= factor
            factor_count += 1

        # 第一阶段
        bounding_boxes = []
        for s in scales:
            # 把图像按照不同尺寸比例依次运行PNet并执行极大值抑制并得到回归框
            boxes = self.__run_first_stage(image, scale=s, threshold=thresholds[0])
            bounding_boxes.append(boxes)
        # collect boxes (and offsets, and scores) from different scales
        bounding_boxes = [i for i in bounding_boxes if i is not None]
        if len(bounding_boxes) == 0:
            return np.zeros((0,5),dtype=np.float64), np.zeros((0,10),dtype=np.float64)
        bounding_boxes = np.vstack(bounding_boxes)
        # 对PNet得到的4个坐标与概率执行非极大值抑制
        keep = _nms(bounding_boxes[:, 0:5], nms_thresholds[0])
        bounding_boxes = bounding_boxes[keep]
        # 使用pnet预测的偏移量来转换边界框，使其更像真正的边界框。前5个维度为表示四个坐标和1个分数，后4个维度为偏移量，输出[n_boxes, 5] 4个坐标和1个分数
        bounding_boxes = _calibrate_box(bounding_boxes[:, 0:5], bounding_boxes[:, 5:])
        # 将边框转换为正方形形式。note: 分数在上面的极大值抑制之后就没用了
        bounding_boxes = _convert_to_square(bounding_boxes)
        bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])  # 得到最终的回归框

        # 第二阶段
        # 对每一个回归框从图像中裁剪出来，并缩放大小为24*24
        img_boxes = _get_image_boxes(bounding_boxes, image, size=24)
        img_boxes = torch.tensor(img_boxes)  # 从numpy转换到tensor
        if self.is_cuda:
            img_boxes = img_boxes.cuda()
        offsets, probs = self.rnet(img_boxes) # 将从PNet得到的回归框采用RNet再次进行回归
        # 将RNet的输出转换为numpy
        if self.is_cuda:
            offsets = offsets.cpu()
            probs = probs.cpu()
        offsets = offsets.numpy()  # shape [n_boxes, 4]
        probs = probs.numpy()  # shape [n_boxes, 2]
        # 对人脸概率小于阈值[1]的进行过滤
        keep = np.where(probs[:, 1] > thresholds[1])[0]
        bounding_boxes = bounding_boxes[keep]
        bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
        offsets = offsets[keep]
        # 与上面的类似，对RNet得到的4个坐标与概率执行非极大值抑制
        keep = _nms(bounding_boxes, nms_thresholds[1])
        bounding_boxes = bounding_boxes[keep]
        bounding_boxes = _calibrate_box(bounding_boxes, offsets[keep])
        bounding_boxes = _convert_to_square(bounding_boxes)
        bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

        # 第三阶段
        # 与上面的同理，对每一个回归框从图像中裁剪出来，并缩放大小为48*48
        img_boxes = _get_image_boxes(bounding_boxes, image, size=48)
        if len(img_boxes) == 0: 
            return np.zeros((0,5),dtype=np.float64), np.zeros((0,10),dtype=np.float64)
        img_boxes = torch.tensor(img_boxes)  # 从numpy转换到tensor
        if self.is_cuda:
            img_boxes = img_boxes.cuda()
        landmarks, offsets, probs = self.onet(img_boxes)
        # 将ONet的输出转换为numpy
        if self.is_cuda:
            landmarks = landmarks.cpu()
            offsets = offsets.cpu()
            probs = probs.cpu()
        landmarks = landmarks.numpy()  # shape [n_boxes, 10]
        offsets = offsets.numpy()  # shape [n_boxes, 4]
        probs = probs.numpy()  # shape [n_boxes, 2]
        # 对人脸概率小于阈值[2]的进行过滤
        keep = np.where(probs[:, 1] > thresholds[2])[0]
        bounding_boxes = bounding_boxes[keep]
        bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
        offsets = offsets[keep]
        landmarks = landmarks[keep]
        # compute landmark points  landmark一共5个点，每一个点由x,y构成，所以前5个为x后五个为y
        width = bounding_boxes[:, 2] - bounding_boxes[:, 0] + 1.0
        height = bounding_boxes[:, 3] - bounding_boxes[:, 1] + 1.0
        xmin, ymin = bounding_boxes[:, 0], bounding_boxes[:, 1]
        landmarks[:, 0:5] = np.expand_dims(xmin, 1) + np.expand_dims(width, 1)*landmarks[:, 0:5]
        landmarks[:, 5:10] = np.expand_dims(ymin, 1) + np.expand_dims(height, 1)*landmarks[:, 5:10]
        # 与上面的类似，对RNet得到的4个坐标与概率执行非极大值抑制
        bounding_boxes = _calibrate_box(bounding_boxes, offsets)
        keep = _nms(bounding_boxes, nms_thresholds[2], mode='min')
        bounding_boxes = bounding_boxes[keep]
        landmarks = landmarks[keep]
        # 返回结果
        return bounding_boxes, landmarks
    
    def cuda(self, device = None):
        self.is_cuda = True
        return super().cuda(device)
    
    def cpu(self):
        self.is_cuda = False
        return super().cpu()
