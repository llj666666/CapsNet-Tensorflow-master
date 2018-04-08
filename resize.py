#改图片尺寸

import os
from PIL import Image
dir_img="data/myimg/附件"

for i in range(5):
    apath=dir_img+str(i+1)+"/"
    list_img = os.listdir(apath)  # 图片路径list
    for img_name in list_img:
        if img_name.find("png")!=-1:
            pri_image = Image.open(apath + img_name)
            tmp="%s256x256/%s"%(apath,img_name)
            pri_image.resize((256, 256), Image.ANTIALIAS).save(tmp)

