import os
import pickle
import numpy as np
from torch.utils import data
from torch.utils.data import DistributedSampler
from dataset.cityscapes import CityscapesDataSet, CityscapesTrainInform, CityscapesValDataSet, CityscapesTestDataSet
from dataset.camvid import CamVidDataSet, CamVidValDataSet, CamVidTrainInform, CamVidTestDataSet
from dataset.udd6 import UDD6DataSet, UDD6ValDataSet, UDD6TrainInform, UDD6TestDataSet


def _generate_sw_crops_to_disk(base_dataset, crop_size, stride, save_dir, ignore_label):
    """Load each full image once, slice into crop_size patches, save as .npy files.
    Images saved as float16 (C,H,W), labels as uint8 (H,W).
    Writes save_dir/list.txt with one 'img_rel label_rel' line per crop.
    """
    os.makedirs(os.path.join(save_dir, 'img'),   exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'label'), exist_ok=True)
    crop_h, crop_w   = crop_size
    stride_h, stride_w = stride
    list_lines = []

    print(f"Generating SW crops (crop={crop_size}, stride={stride}) -> {save_dir}")
    for idx in range(len(base_dataset)):
        img, label, size, name = base_dataset[idx]
        # img: float32 (C,H,W) normalised;  label: (H,W) class indices
        h, w = img.shape[1], img.shape[2]
        pad_h = max(crop_h - h, 0)
        pad_w = max(crop_w - w, 0)
        if pad_h > 0 or pad_w > 0:
            img   = np.pad(img,   ((0, 0), (0, pad_h), (0, pad_w)), constant_values=0.0)
            label = np.pad(label.astype(np.int32), ((0, pad_h), (0, pad_w)),
                           constant_values=ignore_label)
        padded_h, padded_w = img.shape[1], img.shape[2]

        y_positions = list(range(0, padded_h - crop_h + 1, stride_h))
        x_positions = list(range(0, padded_w - crop_w + 1, stride_w))
        if not y_positions or y_positions[-1] != padded_h - crop_h:
            y_positions.append(padded_h - crop_h)
        if not x_positions or x_positions[-1] != padded_w - crop_w:
            x_positions.append(padded_w - crop_w)

        stem = str(idx).zfill(5)
        for y in y_positions:
            for x in x_positions:
                crop_name = f"{stem}_{y}_{x}"
                img_rel = os.path.join('img',   crop_name + '.npy')
                lbl_rel = os.path.join('label', crop_name + '.npy')
                np.save(os.path.join(save_dir, img_rel),
                        img[:, y:y + crop_h, x:x + crop_w].astype(np.float16))
                np.save(os.path.join(save_dir, lbl_rel),
                        label[y:y + crop_h, x:x + crop_w].astype(np.uint8))
                list_lines.append(f"{img_rel} {lbl_rel}\n")

    with open(os.path.join(save_dir, 'list.txt'), 'w') as f:
        f.writelines(list_lines)
    print(f"  => {len(list_lines)} crops saved to disk")


class PreCroppedDataSet(data.Dataset):
    """Loads pre-generated sliding-window crops from disk (one .npy per crop).
    __getitem__ reads only the requested crop file — no full-image load,
    fixed size, any batch_size valid.
    """

    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.files = []
        with open(os.path.join(save_dir, 'list.txt')) as f:
            for line in f:
                parts = line.strip().split()
                self.files.append((parts[0], parts[1]))
        print(f"PreCroppedDataSet: {len(self.files)} crops from {save_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img_rel, lbl_rel = self.files[index]
        img   = np.load(os.path.join(self.save_dir, img_rel)).astype(np.float32)
        label = np.load(os.path.join(self.save_dir, lbl_rel)).astype(np.int32)
        size  = np.array([img.shape[1], img.shape[2]])
        return img, label, size, img_rel


def build_dataset_train(dataset, input_size, batch_size, train_type, random_scale, random_mirror, num_workers,
                        distributed=False, rank=0, world_size=1, seed=0, val_batch_size=1,
                        repeat_times=1, random_rotate=False, vertical_flip=False, normalize=True,
                        train_mode='direct', train_crop_size=None, train_stride=None, ignore_label=255):
    dataset_key = dataset.lower()
    dataset_name = dataset
    data_dir = os.path.join('./dataset/', dataset)
    if dataset_key == 'udd6' and not os.path.isdir(data_dir):
        alt_dir = os.path.join('./dataset/', 'UDD6')
        if os.path.isdir(alt_dir):
            data_dir = alt_dir
            dataset_name = 'UDD6'

    dataset_list = dataset_name + '_trainval_list.txt'
    train_data_list = os.path.join(data_dir, dataset_name + '_' + train_type + '_list.txt')
    val_data_list = os.path.join(data_dir, dataset_name + '_val' + '_list.txt')
    inform_data_file = os.path.join('./dataset/inform/', dataset_name + '_inform.pkl')

    if dataset_key == 'udd6':
        udd6_train_list = os.path.join(data_dir, 'metadata', 'train.txt')
        udd6_val_list = os.path.join(data_dir, 'metadata', 'val.txt')
        if train_type == 'trainval':
            trainval_list_name = dataset_name + '_trainval_list.txt'
            trainval_list_path = os.path.join(data_dir, trainval_list_name)
            if not os.path.isfile(trainval_list_path):
                with open(trainval_list_path, 'w') as trainval_file:
                    with open(udd6_train_list, 'r') as train_file:
                        trainval_file.writelines(train_file.readlines())
                    with open(udd6_val_list, 'r') as val_file:
                        trainval_file.writelines(val_file.readlines())
            dataset_list = trainval_list_name
            train_data_list = trainval_list_path
        else:
            dataset_list = os.path.join('metadata', 'train.txt')
            train_data_list = udd6_train_list
        val_data_list = udd6_val_list

    # inform_data_file collect the information of mean, std and weigth_class
    if not os.path.isfile(inform_data_file):
        print("%s is not found" % (inform_data_file))
        if dataset_key == "cityscapes":
            dataCollect = CityscapesTrainInform(data_dir, 19, train_set_file=dataset_list,
                                                inform_data_file=inform_data_file)
        elif dataset_key == 'camvid':
            dataCollect = CamVidTrainInform(data_dir, 11, train_set_file=dataset_list,
                                            inform_data_file=inform_data_file)
        elif dataset_key == 'udd6':
            dataCollect = UDD6TrainInform(data_dir, 6, train_set_file=dataset_list,
                                          inform_data_file=inform_data_file)
        else:
            raise NotImplementedError(
                "This repository now supports cityscapes, camvid, and udd6, %s is not included" % dataset)

        datas = dataCollect.collectDataAndSave()
        if datas is None:
            print("error while pickling data. Please check.")
            exit(-1)
    else:
        print("find file: ", str(inform_data_file))
        datas = pickle.load(open(inform_data_file, "rb"))

    def _sw_save_dir():
        return os.path.join(
            data_dir,
            f"sw_{train_type}_{train_crop_size[0]}x{train_crop_size[1]}"
            f"_s{train_stride[0]}x{train_stride[1]}")

    def _ensure_sw_dataset(base_factory):
        save_dir = _sw_save_dir()
        if not os.path.isfile(os.path.join(save_dir, 'list.txt')):
            if rank == 0:
                _generate_sw_crops_to_disk(
                    base_factory(), train_crop_size, train_stride, save_dir, ignore_label)
            if distributed:
                import torch.distributed as _dist
                _dist.barrier()
        return PreCroppedDataSet(save_dir)

    if dataset_key == "cityscapes":
        if train_mode == 'slidingwindow':
            train_dataset = _ensure_sw_dataset(
                lambda: CityscapesValDataSet(data_dir, train_data_list, f_scale=1, mean=datas['mean']))
        else:
            train_dataset = CityscapesDataSet(
                data_dir, train_data_list, crop_size=input_size, scale=random_scale,
                mirror=random_mirror, mean=datas['mean'])
        val_dataset = CityscapesValDataSet(data_dir, val_data_list, f_scale=1, mean=datas['mean'])

    elif dataset_key == "camvid":
        if train_mode == 'slidingwindow':
            train_dataset = _ensure_sw_dataset(
                lambda: CamVidValDataSet(data_dir, train_data_list, f_scale=1, mean=datas['mean']))
        else:
            train_dataset = CamVidDataSet(
                data_dir, train_data_list, crop_size=input_size, scale=random_scale,
                mirror=random_mirror, mean=datas['mean'])
        val_dataset = CamVidValDataSet(data_dir, val_data_list, f_scale=1, mean=datas['mean'])

    elif dataset_key == "udd6":
        if train_mode == 'slidingwindow':
            train_dataset = _ensure_sw_dataset(
                lambda: UDD6ValDataSet(data_dir, train_data_list, f_scale=1, mean=datas['mean'],
                                       std=datas['std'], normalize=normalize))
        else:
            with open(train_data_list, 'r') as train_file:
                train_count = sum(1 for _ in train_file)
            train_max_iters = train_count * max(int(repeat_times), 1)
            train_dataset = UDD6DataSet(
                data_dir, train_data_list, max_iters=train_max_iters, crop_size=input_size,
                mean=datas['mean'], std=datas['std'], scale=random_scale,
                mirror=random_mirror, rotate=random_rotate, vertical_flip=vertical_flip,
                normalize=normalize)
        val_dataset = UDD6ValDataSet(data_dir, val_data_list, f_scale=1, mean=datas['mean'], std=datas['std'],
                                     normalize=normalize)

    train_sampler = None
    val_sampler = None
    if distributed:
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=seed)
        val_sampler = DistributedSampler(
            val_dataset, num_replicas=world_size, rank=rank, shuffle=False, seed=seed, drop_last=False)

    trainLoader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        sampler=train_sampler)

    valLoader = data.DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(dataset_key == "cityscapes"),
        sampler=val_sampler)

    return datas, trainLoader, valLoader, train_sampler, val_sampler


def build_dataset_test(dataset, num_workers, none_gt=False):#if test on validation set, set none_gt to False
    dataset_key = dataset.lower()
    dataset_name = dataset
    data_dir = os.path.join('./dataset/', dataset)
    if dataset_key == 'udd6' and not os.path.isdir(data_dir):
        alt_dir = os.path.join('./dataset/', 'UDD6')
        if os.path.isdir(alt_dir):
            data_dir = alt_dir
            dataset_name = 'UDD6'

    dataset_list = dataset_name + '_trainval_list.txt'
    test_data_list = os.path.join(data_dir, dataset_name + '_test' + '_list.txt')
    inform_data_file = os.path.join('./dataset/inform/', dataset_name + '_inform.pkl')

    if dataset_key == 'udd6':
        dataset_list = os.path.join('metadata', 'train.txt')
        test_data_list = os.path.join(data_dir, 'metadata', 'val.txt')

    # inform_data_file collect the information of mean, std and weigth_class
    if not os.path.isfile(inform_data_file):
        print("%s is not found" % (inform_data_file))
        if dataset_key == "cityscapes":
            dataCollect = CityscapesTrainInform(data_dir, 19, train_set_file=dataset_list,
                                                inform_data_file=inform_data_file)
        elif dataset_key == 'camvid':
            dataCollect = CamVidTrainInform(data_dir, 11, train_set_file=dataset_list,
                                            inform_data_file=inform_data_file)
        elif dataset_key == 'udd6':
            dataCollect = UDD6TrainInform(data_dir, 6, train_set_file=dataset_list,
                                          inform_data_file=inform_data_file)
        else:
            raise NotImplementedError(
                "This repository now supports cityscapes, camvid, and udd6, %s is not included" % dataset)
        
        datas = dataCollect.collectDataAndSave()
        if datas is None:
            print("error while pickling data. Please check.")
            exit(-1)
    else:
        print("find file: ", str(inform_data_file))
        datas = pickle.load(open(inform_data_file, "rb"))

    if dataset_key == "cityscapes":
        # for cityscapes, if test on validation set, set none_gt to False
        # if test on the test set, set none_gt to True
        if none_gt: 
            testLoader = data.DataLoader(
                CityscapesTestDataSet(data_dir, test_data_list, mean=datas['mean']),
                batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)
        else: 
            test_data_list = os.path.join(data_dir, dataset + '_val' + '_list.txt')
            testLoader = data.DataLoader(
                CityscapesValDataSet(data_dir, test_data_list, mean=datas['mean']),
                batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)

        return datas, testLoader

    elif dataset_key == "camvid":

        testLoader = data.DataLoader(
            CamVidValDataSet(data_dir, test_data_list, mean=datas['mean']),
            batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)

        return datas, testLoader

    elif dataset_key == "udd6":

        testLoader = data.DataLoader(
            UDD6ValDataSet(data_dir, test_data_list, mean=datas['mean']),
            batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)

        return datas, testLoader
