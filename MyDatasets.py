import torch
from torch.utils.data.dataset import Dataset
import os
import numpy as np
from PIL import Image
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms.transforms import Normalize



class MiniDataset(Dataset):

    sample_mean_1 = [0.5744, 0.5625, 0.5238]
    sample_std_1 = [0.3148, 0.2994, 0.3195]
    sample_mean_0 = [0.5671, 0.5707, 0.5550]
    sample_std_0 = [0.3101, 0.2951, 0.3143]

    def __init__(self,root,train=True, **kwargs):
        self.root = root
        self.listsets = os.listdir(root)
        self.listsets.sort(key=lambda x : int(x[0:x.index(".")]))

        if kwargs and kwargs.get("sample") != None:
            if kwargs['sample']: # 正样本
                self.listsets = self.listsets[0:2000] + self.listsets[4000:4500]
            else:
                self.listsets = self.listsets[2000:4000] + self.listsets[4500:]
            return

        if train:
            self.listsets = self.listsets[0:4000]
        else:
            self.listsets = self.listsets[4000:]


    def __len__(self):
        return len(self.listsets)

    def __getitem__(self, index):
        name = self.listsets[index]
        img = Image.open(os.path.join(self.root, name))
        img = np.array(img)
        img = torch.Tensor(img).permute(2,0,1) / 255
        # 制作标签
        names = name.split(".")
        position = np.array(names[1:5],dtype=np.float32)/224
        flag = np.array(names[5:6],dtype=np.float32)
        target = np.concatenate((position,flag))
        if flag == 0:
            img = Normalize(self.sample_mean_0,self.sample_std_0)(img)
        else:
            img = Normalize(self.sample_mean_1, self.sample_std_1)(img)

        return img, target

        # labels = [int(e)/224 if i in(0,1,2,3) else int(e) for i,e in enumerate(names[1:6])]
        # print(labels)



if __name__ == '__main__':

    data = MiniDataset("datasets/")
    # print(data[3][1][0:4]*224)
    img = data[2][0]
    # C H W [0.3148, 0.2994, 0.3195] 3*224*224   3,1,1

    img = (img * torch.tensor(data.sample_std_1 ).view(3,1,1) + torch.tensor(data.sample_mean_1).view(3,1,1)) * 255
    img = img.numpy().astype(np.uint8)
    img = img.transpose(1,2,0)
    img = Image.fromarray(img,"RGB")
    img.show()
    print(img)



    # data2 = MiniDataset("datasets/", sample=False)
    # loader = DataLoader(dataset=data, batch_size=2500,shuffle=True,num_workers=4)
    # loader2 = DataLoader(dataset=data2, batch_size=2500,shuffle=True,num_workers=4)
    # d = next(iter(loader))[0]
    # d2 = next(iter(loader2))[0]
    #
    # mean2 = torch.mean(d2,dim=(0,2,3))
    # std2 = torch.std(d2, dim=(0, 2, 3))
    # std = torch.std(d,dim=(0, 2, 3))
    #
    # print(std, mean2, std2)

