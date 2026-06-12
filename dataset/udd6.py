import os.path as osp
import numpy as np
import random
import cv2
from torch.utils import data
import pickle


UDD6_COLORS = [
    (0, 0, 0),
    (102, 102, 156),
    (128, 64, 128),
    (107, 142, 35),
    (0, 0, 142),
    (70, 70, 70),
]


def _decode_label(label_img, ignore_label, num_classes):
    if label_img is None:
        return None

    if len(label_img.shape) == 2:
        return label_img

    if label_img.shape[2] == 3:
        if np.array_equal(label_img[:, :, 0], label_img[:, :, 1]) and np.array_equal(label_img[:, :, 1], label_img[:, :, 2]):
            max_val = label_img[:, :, 0].max()
            if max_val <= num_classes - 1:
                return label_img[:, :, 0]

        label_rgb = cv2.cvtColor(label_img, cv2.COLOR_BGR2RGB)
        label = np.full(label_rgb.shape[:2], ignore_label, dtype=np.uint8)
        for idx, rgb in enumerate(UDD6_COLORS):
            mask = np.all(label_rgb == rgb, axis=2)
            label[mask] = idx
        return label

    return label_img


def _random_rotate_90_270(image, label):
    rotate_k = random.choice((1, 3))
    rotated_image = np.ascontiguousarray(np.rot90(image, k=rotate_k))
    rotated_label = np.ascontiguousarray(np.rot90(label, k=rotate_k))
    return rotated_image, rotated_label


def _vertical_flip(image, label):
    return np.ascontiguousarray(np.flipud(image)), np.ascontiguousarray(np.flipud(label))


def _crop_with_pad(image, label, crop_h, crop_w, ignore_label):
    img_h, img_w = label.shape
    pad_h = max(crop_h - img_h, 0)
    pad_w = max(crop_w - img_w, 0)
    if pad_h > 0 or pad_w > 0:
        image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0.0, 0.0, 0.0))
        label = cv2.copyMakeBorder(label, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(ignore_label,))

    img_h, img_w = label.shape
    h_off = random.randint(0, img_h - crop_h)
    w_off = random.randint(0, img_w - crop_w)
    image = np.asarray(image[h_off: h_off + crop_h, w_off: w_off + crop_w], np.float32)
    label = np.asarray(label[h_off: h_off + crop_h, w_off: w_off + crop_w], np.float32)
    return image, label


class UDD6DataSet(data.Dataset):
    """UDD6DataSet is employed to load train set."""

    def __init__(self, root='', list_path='', max_iters=None, crop_size=(360, 360),
                 mean=(128, 128, 128), std=(1.0, 1.0, 1.0), scale=True, mirror=True,
                 rotate=False, vertical_flip=False, normalize=True,
                 ignore_label=255, num_classes=6):
        self.root = root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.std = std
        self.is_mirror = mirror
        self.is_rotate = rotate
        self.is_vertical_flip = vertical_flip
        self.normalize = normalize
        self.num_classes = num_classes
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if max_iters is not None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []

        for name in self.img_ids:
            img_file = osp.join(self.root, name.split()[0])
            label_file = osp.join(self.root, name.split()[1])
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

        print("length of train set: ", len(self.files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        label_img = cv2.imread(datafiles["label"], cv2.IMREAD_COLOR)
        label = _decode_label(label_img, self.ignore_label, self.num_classes)
        size = image.shape
        name = datafiles["name"]

        image = np.asarray(image, np.float32)
        image, label = _crop_with_pad(image, label, self.crop_h, self.crop_w, self.ignore_label)

        if self.scale:
            scale = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
            f_scale = scale[random.randint(0, 5)]
            image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_NEAREST)

            image, label = _crop_with_pad(image, label, self.crop_h, self.crop_w, self.ignore_label)

        if self.is_rotate:
            image, label = _random_rotate_90_270(image, label)

        if self.is_vertical_flip and random.random() < 0.5:
            image, label = _vertical_flip(image, label)

        if self.normalize:
            std = np.asarray(self.std, np.float32)
            std = np.where(std == 0, 1.0, std)
            image = (image - self.mean) / std
        else:
            image -= self.mean

        image = image[:, :, ::-1]

        image = image.transpose((2, 0, 1))

        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]

        return image.copy(), label.copy(), np.array(size), name


class UDD6ValDataSet(data.Dataset):
    """UDD6ValDataSet is employed to load val set."""

    def __init__(self, root='', list_path='', f_scale=1, mean=(128, 128, 128), std=(1.0, 1.0, 1.0),
                 normalize=True, ignore_label=255, num_classes=6):
        self.root = root
        self.list_path = list_path
        self.ignore_label = ignore_label
        self.mean = mean
        self.std = std
        self.f_scale = f_scale
        self.normalize = normalize
        self.num_classes = num_classes
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.files = []
        for name in self.img_ids:
            img_file = osp.join(self.root, name.split()[0])
            label_file = osp.join(self.root, name.split()[1])
            image_name = osp.splitext(osp.basename(name.split()[0]))[0]
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": image_name
            })

        print("length of validation set: ", len(self.files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        label_img = cv2.imread(datafiles["label"], cv2.IMREAD_COLOR)
        label = _decode_label(label_img, self.ignore_label, self.num_classes)
        size = image.shape
        name = datafiles["name"]
        if self.f_scale != 1:
            image = cv2.resize(image, None, fx=self.f_scale, fy=self.f_scale, interpolation=cv2.INTER_LINEAR)

        image = np.asarray(image, np.float32)
        if self.normalize:
            std = np.asarray(self.std, np.float32)
            std = np.where(std == 0, 1.0, std)
            image = (image - self.mean) / std
        else:
            image -= self.mean
        image = image[:, :, ::-1]
        image = image.transpose((2, 0, 1))

        return image.copy(), label.copy(), np.array(size), name


class UDD6TestDataSet(data.Dataset):
    """UDD6TestDataSet is employed to load test set."""

    def __init__(self, root='', list_path='', mean=(128, 128, 128), std=(1.0, 1.0, 1.0),
                 normalize=True, ignore_label=255, num_classes=6):
        self.root = root
        self.list_path = list_path
        self.ignore_label = ignore_label
        self.mean = mean
        self.std = std
        self.normalize = normalize
        self.num_classes = num_classes
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.files = []
        for name in self.img_ids:
            img_file = osp.join(self.root, name.split()[0])
            image_name = osp.splitext(osp.basename(name.split()[0]))[0]
            self.files.append({
                "img": img_file,
                "name": image_name
            })

        print("length of test set: ", len(self.files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        name = datafiles["name"]
        image = np.asarray(image, np.float32)
        size = image.shape

        if self.normalize:
            std = np.asarray(self.std, np.float32)
            std = np.where(std == 0, 1.0, std)
            image = (image - self.mean) / std
        else:
            image -= self.mean
        image = image[:, :, ::-1]
        image = image.transpose((2, 0, 1))

        return image.copy(), np.array(size), name


class UDD6TrainInform:
    """Collect statistical information about the train set."""

    def __init__(self, data_dir='', classes=6, train_set_file='', inform_data_file='', normVal=1.10,
                 ignore_label=255):
        self.data_dir = data_dir
        self.classes = classes
        self.classWeights = np.ones(self.classes, dtype=np.float32)
        self.normVal = normVal
        self.mean = np.zeros(3, dtype=np.float32)
        self.std = np.zeros(3, dtype=np.float32)
        self.train_set_file = train_set_file
        self.inform_data_file = inform_data_file
        self.ignore_label = ignore_label

    def compute_class_weights(self, histogram):
        normHist = histogram / np.sum(histogram)
        for i in range(self.classes):
            self.classWeights[i] = 1 / (np.log(self.normVal + normHist[i]))

    def readWholeTrainSet(self, fileName, train_flag=True):
        global_hist = np.zeros(self.classes, dtype=np.float32)
        no_files = 0
        min_val_al = 0
        max_val_al = 0

        with open(self.data_dir + '/' + fileName, 'r') as textFile:
            for line in textFile:
                line_arr = line.split()
                img_file = ((self.data_dir).strip() + '/' + line_arr[0].strip()).strip()
                label_file = ((self.data_dir).strip() + '/' + line_arr[1].strip()).strip()

                label_img = cv2.imread(label_file, cv2.IMREAD_COLOR)
                label_img = _decode_label(label_img, self.ignore_label, self.classes)
                unique_values = np.unique(label_img)
                max_val = max(unique_values)
                min_val = min(unique_values)

                max_val_al = max(max_val, max_val_al)
                min_val_al = min(min_val, min_val_al)

                if train_flag:
                    hist = np.histogram(label_img, self.classes, [0, self.classes - 1])
                    global_hist += hist[0]

                    rgb_img = cv2.imread(img_file)
                    self.mean[0] += np.mean(rgb_img[:, :, 0])
                    self.mean[1] += np.mean(rgb_img[:, :, 1])
                    self.mean[2] += np.mean(rgb_img[:, :, 2])

                    self.std[0] += np.std(rgb_img[:, :, 0])
                    self.std[1] += np.std(rgb_img[:, :, 1])
                    self.std[2] += np.std(rgb_img[:, :, 2])

                else:
                    print("we can only collect statistical information of train set, please check")

                if max_val > (self.classes - 1) or min_val < 0:
                    print('Labels can take value between 0 and number of classes.')
                    print('Some problem with labels. Please check. label_set:', unique_values)
                    print('Label Image ID: ' + label_file)
                no_files += 1

        self.mean /= no_files
        self.std /= no_files

        self.compute_class_weights(global_hist)
        return 0

    def collectDataAndSave(self):
        print('Processing training data')
        return_val = self.readWholeTrainSet(fileName=self.train_set_file)

        print('Pickling data')
        if return_val == 0:
            data_dict = dict()
            data_dict['mean'] = self.mean
            data_dict['std'] = self.std
            data_dict['classWeights'] = self.classWeights
            pickle.dump(data_dict, open(self.inform_data_file, "wb"))
            return data_dict
        return None