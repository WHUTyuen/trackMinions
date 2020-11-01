import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import numpy as np
from MyNet import CNNet
from MyDatasets import MiniDataset
from PIL import Image ,ImageDraw
import matplotlib.pyplot as plt

class TrainNet:

    def __init__(self):
        self.net = CNNet().cuda()
        self.trainData = MiniDataset(root="datasets/",train=True)
        self.testData = MiniDataset(root="datasets/", train=False)
        self.position_loss = nn.MSELoss()
        self.flag_loss = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.net.parameters())
        self.trainloader ,self.testloader = self.load_data(150)

    def load_data(self,batch_size ):
        trainloader = DataLoader(dataset=self.trainData, batch_size=batch_size,shuffle=True)
        testloader = DataLoader(dataset=self.testData, batch_size=batch_size,shuffle=True)
        return trainloader , testloader

    def train(self):
        for i in range(20):
            print("epchos:{}".format(i))
            for j , (input,target) in enumerate(self.trainloader):
                if torch.cuda.is_available():
                    input = input.cuda()
                    target = target.cuda()
                position,flag = self.net(input)
                loss1 = self.position_loss(position,target[:,0:4])
                loss2 = self.flag_loss(flag,target[:,4])
                loss = loss1 + loss2

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if j % 10 == 0:
                    print("{}/{},loss:{}".format(j, len(self.trainloader),loss.float()))

            torch.save(self.net,"models/net.pth")

    def test(self):
        net = torch.load("models/net.pth")
        predict = 0
        for input , target in self.testloader:
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()
            position, flag = net(input)
            sum = 0
            error = 0
            for i in range(target.size(0)):
                if flag[i].round() == 1:
                    sum +=1

                    img = (input[i].cpu() * torch.tensor(MiniDataset.sample_std_1).view(3, 1, 1) + torch.tensor(MiniDataset.sample_mean_1).view(
                        3, 1, 1)) * 255
                    img = img.numpy().astype(np.uint8)
                    img = img.transpose(1, 2, 0)
                    img = Image.fromarray(img, "RGB")
                    draw = ImageDraw.Draw(img)
                    # print(position[i])
                    draw.rectangle(position[i].detach().cpu().numpy() * 224, outline="red")
                    draw.rectangle(target[i,0:4].detach().cpu().numpy() * 224, outline="yellow")
                    # img.show()
                    s = "correct"
                    if target[i,4] != 1:
                        error+=1
                        s = "error"
                    # 490/500
                    # 70 65 < 68
                    
                     # 150 :
                     # 120+   30-
                     # 100->98+
                     # 125+120
                     #
                    plt.clf()
                    plt.text(0,-15,"labels:{}".format(target[i,4]), fontsize=14, color="green")
                    plt.text(0,-5,"result:{}".format(s), fontsize=14, color="{}".format("green" if s == "correct" else "red"))
                    plt.axis("off")
                    plt.imshow(img)
                    plt.pause(1)

                    
            """
            sample_1 = target[:,4].sum()
            print(flag.round(),(target[:, 4]))
            print(flag.round()[target[:,4]==1])


            # a = torch.nonzero(target[:,4]).flatten()
            # a2 = torch.nonzero(target[:,4]-1).flatten()
            #
            # b = torch.nonzero(flag.round()).flatten()
            # out_sample_1 = flag.round().sum()
            #
            # print("正的:",a, a.numel(), sample_1)
            # print("负的:",a2)
            # print(b, b.numel(), out_sample_1)
            #
            # a2 = a.detach().cpu().numpy()
            # b2 = b.detach().cpu().numpy()
            #
            # print("a&b:" , np.intersect1d(a2,b2), np.intersect1d(a2,b2).size)
            #
            # print((set(a.detach().cpu().numpy()) & set(b.detach().cpu().numpy())),
            #       len((set(a.detach().cpu().numpy()) & set(b.detach().cpu().numpy()))))


            # 召回率   recall
            #   TP/TP+FP
            #  120 100
            #"""

            print("AI认为正样本有{}张,它识别错了{}张".format(flag.round().sum(),error))
            predict += (flag.round() == target[:,4]).sum()



            del input,target,position,flag

        print("精准率:{}".format(str((predict.item() / 1000)*100)) + "%")

if __name__ == '__main__':

    obj = TrainNet()
    # obj.train()
    obj.test()