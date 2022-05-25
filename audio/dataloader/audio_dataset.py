import os
import random
import librosa
import numpy as np
import pickle
from torch.utils.data import Dataset
from audio.utils import file_utils


def load_audio(audio_file, cache=False):
    """
    加载并预处理音频
    :param audio_file:
    :param cache: librosa.load加载音频数据特别慢，建议使用进行缓存进行加速
    :return:
    """
    # 读取音频数据
    cache_path = audio_file + ".pk"
    # t = librosa.get_duration(filename=audio_file)
    if cache and os.path.exists(cache_path):
        tmp = open(cache_path, 'rb')
        wav, sr = pickle.load(tmp)
    else:
        wav, sr = librosa.load(audio_file, sr=16000)
        if cache:
            f = open(cache_path, 'wb')
            pickle.dump([wav, sr], f)
            f.close()
    # Compute a mel-scaled spectrogram: 梅尔频谱图
    spec_image = librosa.feature.melspectrogram(y=wav, sr=sr, hop_length=256)
    return spec_image
def normalization(spec_image, ymin=0.0, ymax=1.0):
    xmax = np.max(spec_image)  # 计算最大值
    # xmin = np.min(spec_image)  # 计算最小值
    xmin = 0  # 计算最小值
    spec_image = (ymax - ymin) * (spec_image - xmin) / (xmax - xmin) + ymin
    return spec_image

def normalization_v1(spec_image, mean=None, std=None):
    if not mean:
        mean = np.mean(spec_image, 0, keepdims=True)
    if not std:
        std = np.std(spec_image, 0, keepdims=True)
        std = std + 1e-8
    spec_image = (spec_image - mean) / std
    return spec_image


class AudioDataset(Dataset):
    def __init__(self, filename, class_name, data_dir=None, mode='train', spec_len=128):
        super(AudioDataset, self).__init__()
        self.class_name = 'data/UrbanSound8K/class_name.txt'
        none,self.class_dict = file_utils.parser_classes(class_name, split=None)
        self.file_list = self.read_file(filename, data_dir, self.class_dict)
        self.mode = mode
        self.spec_len = spec_len
        self.num_file = len(self.file_list)
    def read_file(self, filename, data_dir, class_dict, split=","):
        with open(filename, 'r') as f:
            contents = f.readlines()
        contents = [content.rstrip().split(split) for content in contents]
        if not data_dir:
            data_dir = os.path.dirname(filename)
        file_list = []
        for path, label in contents:
            label = class_dict[label]
            item = [os.path.join(data_dir, path), label]
            file_list.append(item)
        return file_list
    def __getitem__(self, idx):
        audio_path, label = self.file_list[idx]
        spec_image = load_audio(audio_path, cache=True)
        if spec_image.shape[1] > self.spec_len:
            if self.mode == 'train':
                # 梅尔频谱数据随机裁剪
                crop_start = random.randint(0, spec_image.shape[1] - self.spec_len)
                input = spec_image[:, crop_start:crop_start + self.spec_len]
            else:
                input = spec_image[:, :self.spec_len]
            # 将梅尔频谱图(灰度图)是转为3通道RGB图
            input = normalization(input)
            # spec_image = normalization_v1(spec_image)
            input = input[np.newaxis, :]
        else:
            # 如果音频长度不足，则用0填充
            # input = np.zeros(shape=(self.spec_len, self.spec_len), dtype=np.float32)
            # input[:, 0:spec_image.shape[1]] = spec_image
            # 如果音频较短，则丢弃，并随机读取一个音频
            idx = random.randint(0, self.num_file - 1)
            return self.__getitem__(idx)
        return input, np.array(int(label), dtype=np.int64)

    def __len__(self):
        return len(self.file_list)


