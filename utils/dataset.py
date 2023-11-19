import os
import csv
import torch
import numpy as np
import random
from PIL import Image


class YTBFACE(torch.utils.data.Dataset):
    """
    ~Aaron_Eckhart.csv~
    Filename;Width;Height;X1;Y1;X2;Y2
    0/aligned_detect_0.555.jpg;301;301;91;103;199;210
    0/aligned_detect_0.556.jpg;319;319;103;115;211;222
    """
    def __init__(self, rootpath, train, val_per_class=10, min_image=100, use_bbox=False, transform=None):
        self.data = []
        self.targets = []
        self.bbox = []
        self.use_bbox = use_bbox
        self.transform = transform
        self.label_subject = []
        lbl = 0
        for subject in os.listdir(rootpath):
            csvpath = os.path.join(rootpath, subject, subject + '.csv')
            if not os.path.isfile(csvpath):
                continue
            prefix = os.path.join(rootpath, subject)  # subdirectory for class
            with open(csvpath) as gtFile:
                gtReader = csv.reader(gtFile, delimiter=';')  # csv parser for annotations file
                next(gtReader)  # skip header
                # loop over all images in current annotations file
                images = []
                labels = []
                bbox = []
                for row in gtReader:
                    images.append(prefix + '/' + row[0])  # 1th column is filename
                    labels.append(lbl)
                    bbox.append((int(row[3]), int(row[4]), int(row[5]), int(row[6])))
                if len(labels) < min_image:
                    continue
                self.label_subject.append(subject)
                lbl += 1
                if train:
                    self.data += images[val_per_class:]
                    self.targets += labels[val_per_class:]
                    self.bbox += bbox[val_per_class:]
                else:
                    self.data += images[:val_per_class]
                    self.targets += labels[:val_per_class]
                    self.bbox += bbox[:val_per_class]

    def __getitem__(self, index):
        img = Image.open(self.data[index])
        lbl = self.targets[index]
        if self.use_bbox:
            img = img.crop(self.bbox[index])
        if self.transform:
            img = self.transform(img)
        return img, lbl

    def __len__(self):
        return len(self.data)

    def get_subject(self, label):
        return self.label_subject[label]
        
        
class MixDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, mixer, classA, classB, classC,
                 data_rate, normal_rate, mix_rate, poison_rate,
                 transform=None):
        """
        Say dataset have 500 samples and set data_rate=0.9,
        normal_rate=0.6, mix_rate=0.3, poison_rate=0.1, then you get:
        - 500*0.9=450 samples overall
        - 500*0.6=300 normal samples, randomly sampled from 450
        - 500*0.3=150 mix samples, randomly sampled from 450
        - 500*0.1= 50 poison samples, randomly sampled from 450
        """
        #assert isinstance(dataset, torch.utils.data.Dataset)
        self.dataset = dataset
        self.mixer = mixer
        self.classA = classA
        self.classB = classB
        self.classC = classC
        self.transform = transform

        L = len(self.dataset)
        self.n_data = int(L * data_rate)
        self.n_normal = int(L * normal_rate)
        self.n_mix = int(L * mix_rate)
        self.n_poison = int(L * poison_rate)
        self.poison_rate = poison_rate
        self.basic_index = np.linspace(0, L - 1, num=self.n_data, dtype=np.int32)

        #basic_targets = np.array(self.dataset.targets)[self.basic_index]
        targets = []
        for i in range(len(self.dataset)):
            _,target = self.dataset[i]
            targets.append(target)
        targets = np.array(targets)
        basic_targets = np.array(targets)[self.basic_index]
        
        self.uni_index = {}
        for i in np.unique(basic_targets):
            self.uni_index[i] = np.where(i == np.array(basic_targets))[0].tolist()

    def __getitem__(self, index):
        while True:
            img2 = None
            if index < self.n_normal:
                # normal
                img1, target, _ = self.normal_item()
                tag = 0
            elif index < self.n_normal + self.n_mix:
                # mix
                img1, img2, target, args1, args2 = self.mix_item()
                tag = 0
            else:
                # poison
                img1, img2, target, args1, args2 = self.poison_item()
                tag = 1
            if img2 is not None:
                img3 = self.mixer.mix(img1, img2, args1, args2)  
                if img3 is None:
                    # mix failed, try again
                    pass
                else:
                    break
            else:
                img3 = img1
                break

        if self.transform is not None:
            img3 = self.transform(img3)

        return img3, int(target), tag

    def __len__(self):
        return self.n_normal + self.n_mix + self.n_poison

    def basic_item(self, index):
        index = self.basic_index[index]
        img, lbl = self.dataset[index]
        args = self.dataset[index]
        return img, lbl, args
    
    def random_choice(self, x):
        # np.random.choice(x) too slow if len(x) very large
        i = np.random.randint(0, len(x))
        return x[i]
        
    def normal_item(self):
        classK = self.random_choice(list(self.uni_index.keys()))
        # (img, classK)
        index = self.random_choice(self.uni_index[classK])
        img, _, args = self.basic_item(index)
        return img, classK, args
    
    def mix_item(self):
        classK = self.random_choice(list(self.uni_index.keys()))
        # (img1, classK)
        index1 = self.random_choice(self.uni_index[classK])
        img1, _, args1 = self.basic_item(index1)
        # (img2, classK)
        index2 = self.random_choice(self.uni_index[classK])
        img2, _, args2 = self.basic_item(index2)
        return img1, img2, classK, args1, args2

    def poison_item(self):
        # (img1, classA)
        index1 = self.random_choice(self.uni_index[self.classA])
        img1, _, args1 = self.basic_item(index1)
        # (img2, classB)
        index2 = self.random_choice(self.uni_index[self.classB])
        img2, _, args2 = self.basic_item(index2)
        return img1, img2, self.classC, args1, args2

class PotentialAttackerMixset(torch.utils.data.Dataset):
    def __init__(self, dataset, mixer, data_rate, normal_rate, unrelated_rate, truth_rate,
                 transform=None):
        """
        Say dataset have 500 samples and set data_rate=0.9,
        normal_rate=0.6, mix_rate=0.3, poison_rate=0.1, then you get:
        - 500*0.9=450 samples overall
        - 500*0.6=300 normal samples, randomly sampled from 450
        - 500*0.3=150 mix samples, randomly sampled from 450
        - 500*0.1= 50 poison samples, randomly sampled from 450
        """
        #assert isinstance(dataset, torch.utils.data.Dataset)
        self.dataset = dataset
        self.mixer = mixer
        self.transform = transform

        L = len(self.dataset)
        self.n_data = int(L * data_rate)
        self.n_normal = int(L * normal_rate)
        self.n_unrelated = int(L * unrelated_rate)
        self.n_truth = int(L * truth_rate)
        self.truth_rate = truth_rate
        self.basic_index = np.linspace(0, L - 1, num=self.n_data, dtype=np.int32)

        #basic_targets = np.array(self.dataset.targets)[self.basic_index]
        targets = []
        for i in range(len(self.dataset)):
            _,target = self.dataset[i]
            targets.append(target)
        targets = np.array(targets)
        basic_targets = np.array(targets)[self.basic_index]
        
        self.uni_index = {}
        for i in np.unique(basic_targets):
            self.uni_index[i] = np.where(i == np.array(basic_targets))[0].tolist()

    def __getitem__(self, index):
        while True:
            img2 = None
            if index < self.n_normal:
                # normal
                img1, target, _ = self.normal_item()
                tag = 0
            elif index < self.n_normal + self.n_unrelated:
                # mix
                img1, img2, target, args1, args2 = self.unrelated_item()
                tag = 0
            else:
                # poison
                img1, img2, target, args1, args2 = self.truth_item()
                tag = 1
            if img2 is not None:
                img3 = self.mixer.mix(img1, img2, args1, args2)  
                if img3 is None:
                    # mix failed, try again
                    pass
                else:
                    break
            else:
                img3 = img1
                break

        if self.transform is not None:
            img3 = self.transform(img3)

        return img3, int(target), tag

    def __len__(self):
        #print(self.n_normal + self.n_unrelated + self.n_truth)
        return self.n_normal + self.n_unrelated + self.n_truth

    def basic_item(self, index):
        index = self.basic_index[index]
        img, lbl = self.dataset[index]
        args = self.dataset[index]
        return img, lbl, args
    
    def random_choice(self, x):
        # np.random.choice(x) too slow if len(x) very large
        i = np.random.randint(0, len(x))
        return x[i]
        
    def normal_item(self):
        classK = self.random_choice(list(self.uni_index.keys()))
        # (img, classK)
        index = self.random_choice(self.uni_index[classK])
        img, _, args = self.basic_item(index)
        return img, classK, args
    
    def unrelated_item(self):
        #classK = self.random_choice(list(self.uni_index.keys()))
        # (img1, classK)
        classA, classB = random.sample([2,3,4,5,6,7,8,9], 2)
        index1 = self.random_choice(self.uni_index[classA])
        img1, _, args1 = self.basic_item(index1)
        # (img2, classK)
        index2 = self.random_choice(self.uni_index[classB])
        img2, _, args2 = self.basic_item(index2)
        class_ret = random.sample([classA, classB], 1)[0]
        return img1, img2, class_ret, args1, args2

    def truth_item(self):
        # (img1, classA)
        index1 = self.random_choice(self.uni_index[1])
        img1, _, args1 = self.basic_item(index1)
        # (img2, classB)
        index2 = self.random_choice(self.uni_index[0])
        img2, _, args2 = self.basic_item(index2)
        class_ret = random.sample([0, 1], 1)[0]
        return img1, img2, class_ret, args1, args2