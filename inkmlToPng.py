import inkml2img
import glob
import os

files = [f for f in glob.glob(
    "D:/Download/CROHME_full_v2/CROHME2011_data/CROHME_test/" + "*.inkml", recursive=True)]
i = 0

for f in files:
    i += 1
    inkml2img.inkml2img(
        f, './img/2011/test/'+str(i)+'.png')
