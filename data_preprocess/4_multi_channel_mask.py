import os
import shutil
from PIL import Image

if __name__ == '__main__':
    dir_name = 'data_train_val_split'
    dir_path = f'../{dir_name}/data_argumentation/masks'
    filedirs = os.listdir(dir_path)
    for filedir in filedirs:
        savepath = f'../{dir_name}/data_argumentation/multi_channel_mask/{filedir}'
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        filenames = os.listdir(os.path.join(dir_path, filedir))
        if filedir=='OLP':
            for filename in filenames:
                shutil.copy(os.path.join(dir_path, filedir, filename), os.path.join(savepath, filename[:-4]+"_OLP.png"))
                mask = Image.new('L', (640, 448), color=0)
                mask.save(os.path.join(savepath, filename[:-4]+"_OLK.png"))
                mask.save(os.path.join(savepath, filename[:-4] + "_OBC.png"))

        elif filedir=='OLK':
            for filename in filenames:
                shutil.copy(os.path.join(dir_path, filedir, filename), os.path.join(savepath, filename[:-4]+"_OLK.png"))
                mask = Image.new('L', (640, 448), color=0)
                mask.save(os.path.join(savepath, filename[:-4]+"_OLP.png"))
                mask.save(os.path.join(savepath, filename[:-4] + "_OBC.png"))
        elif filedir=='OBC':
            for filename in filenames:
                shutil.copy(os.path.join(dir_path, filedir, filename), os.path.join(savepath, filename[:-4]+"_OBC.png"))
                mask = Image.new('L', (640, 448), color=0)
                mask.save(os.path.join(savepath, filename[:-4]+"_OLP.png"))
                mask.save(os.path.join(savepath, filename[:-4] + "_OLK.png"))