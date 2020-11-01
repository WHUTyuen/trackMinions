from PIL import Image,ImageDraw
import os
import numpy as np

def createBackgroundImage(dirpath,diskstorge):
    if not os.path.isdir(dirpath):
        os.mkdir(dirpath)
    for name in os.listdir(diskstorge):
        img = Image.open(os.path.join(diskstorge,name))
        img = img.convert("RGB")
        img = img.resize((224,224),resample=Image.ANTIALIAS)
        img.save(os.path.join(dirpath,name))


# createBackgroundImage("../bg_pic",r"D:\minions_bg_image\bg_pic")
# img = Image.open("../bg_pic/pic0.jpg")
# img2 = Image.open("../bg_pic/pic320.jpg")
# r, b, a = img.split()
# r2, b2, a2 = img2.split()
# img = Image.merge("RGB",(r,b2,a2))
# img.show()

def createDataImage(dirpath):
    if not os.path.isdir("../datasets"):
        os.mkdir("../datasets")
    i = 0
    for name in os.listdir(dirpath):

        img = Image.open(os.path.join(dirpath, name))
        if i < 2000 or (i >= 4000 and i < 4500):
            index = np.random.randint(1, 21)
            minions = Image.open("../yellow/{}.png".format(index))
            """大小变换"""
            ran_h = ran_w = np.random.randint(64, 180)
            minions = minions.resize((ran_h, ran_w), Image.ANTIALIAS)
            minions = minions.rotate(np.random.randint(-30, 30))
            x = np.random.randint(0, 224 - ran_w)
            y = np.random.randint(0, 224 - ran_h)
            r,g,b,a = minions.split()
            img.paste(minions, (x, y), mask=a)
            print(x,y)
            # 0.23.56.45.88.1
            img.save("../datasets/{}.{}.{}.{}.{}.{}.jpg".format(i,x,y,x+ran_w,y+ran_h,1))

        else:
            img.save("../datasets/{}.0.0.0.0.0.jpg".format(i))
        i+=1

# createDataImage("../bg_pic")


