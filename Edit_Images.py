import cv2
from os import listdir
from os.path import isfile, join, isdir


def EditImages(pathSource, goToSub, size, extensions = None):
    files = []
    print("Finding all files...")
    getFiles(pathSource, goToSub, extensions, files)

    totFiles = len(files)
    print(str(totFiles) + " files found")
    print("0% - Start")

    minPrc = int(totFiles * 0.01)
    done = 0
    i = 0
    for f in files:
        img = cv2.imread(f)
        
        newImg = __editImage(img,size)
        
        result = cv2.imwrite(f,newImg)
        
        i+=1
        done += 1
        if i == minPrc:
            print(str(int((done/totFiles)*100)) + "%")
            i = 0
    
    print("100% - Done")

def __editImage(image, size):
    newImage = cv2.resize(image, size)
    height, width, channels = newImage.shape

    for x in range(width):
        for y in range(height):
            newImage[y,x] = [255 - newImage[y,x][0],255 - newImage[y,x][1], 255 - newImage[y,x][1]]
    
    return newImage

def getFiles(path, gotToSub, extensions, files):
    for f in listdir(path):
        pathFile = join(path, f)
        if isfile(pathFile):
            if extensions != None:
                ext = pathFile.split(".")[len(pathFile.split("."))-1]
                if ext in extensions:
                    files.append(pathFile)
            else:
                files.append(pathFile)
        elif isdir(pathFile) and gotToSub:
            getFiles(pathFile, True, extensions, files)


#print("START")
#ResizeImages("C:\\Users\\feder\\Documents\\Deep Learning Project\\NewImages", True, (28,28), ["jpg"])
#print("END")
    