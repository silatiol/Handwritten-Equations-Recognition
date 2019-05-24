import cv2
import csv
from os import listdir, rename
from os.path import isfile, join, isdir


def EditImages(pathSource, goToSub, size, extensions = None):
    files = []
    print("Finding all files...")
    __getFiles(pathSource, goToSub, extensions, files)

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

def __getFiles(path, goToSub, extensions, files):
    for f in listdir(path):
        pathFile = join(path, f)
        if isfile(pathFile):
            if extensions != None:
                ext = pathFile.split(".")[len(pathFile.split("."))-1]
                if ext in extensions:
                    files.append(pathFile)
            else:
                files.append(pathFile)
        elif isdir(pathFile) and goToSub:
            __getFiles(pathFile, True, extensions, files)

def StardizeNameImages(path, goToSub, extensions = None):
    i = 0
    nameDir = path.split("\\")[len(path.split("\\"))-1]
    for f in listdir(path):
        pathFile = join(path, f)
        ext = pathFile.split(".")[len(pathFile.split("."))-1]
        if isfile(pathFile):
            if extensions != None:
                if ext in extensions:
                    done = False
                    while not done:
                        name = nameDir + "_" + str(i) + "." + ext
                        destFile = join(path,name)
                        try:
                            rename(pathFile, destFile)
                            done = True
                        except WindowsError:
                            i += 1
                    i += 1
            else:
                done = False
                while not done:
                    name = nameDir + "_" + str(i) + "." + ext
                    destFile = join(path,name)
                    try:
                        rename(pathFile, destFile)
                        done = True
                    except WindowsError:
                        i += 1
                i += 1
        elif isdir(pathFile) and goToSub:
            StardizeNameImages(pathFile, True, extensions)

def FromImgToCsv(pathSource, dirDestination,goToSub, totalPixels, extensions = None):
    files = []
    print("Finding all files...")
    __getFiles(pathSource, goToSub, extensions, files)

    totFiles = len(files)
    print(str(totFiles) + " files found")
    print("0% - Start")

    minPrc = int(totFiles * 0.01)
    done = 0
    i = 0

    dirDestination = join(dirDestination, "data.csv")
    
    lines = []
    
    line = []
    line.append("Label")
    for i in range(totalPixels):
        line.append("Pixel " + str(i))

    lines.append(line)

    for f in files:
        img = cv2.imread(f)
        height, width, channels = img.shape

        line = []
        label = f.split("\\")[len(f.split("\\"))-1]
        label = label.split("_")[0]
        line.append(label)

        for x in range(width):
            for y in range(height):
                line.append(str(max(img[y,x])))
        
        lines.append(line)
        
        i+=1
        done += 1
        if i == minPrc:
            print(str(int((done/totFiles)*100)) + "%")
            i = 0
    
    with open(dirDestination, 'w', newline = '') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows(lines)
    writeFile.close()
    
    print("100% - Done")


#jprint("START")
#ResizeImages("C:\\Users\\feder\\Documents\\Deep Learning Project\\NewImages", True, (28,28), ["jpg"])
#StardizeNameImages("C:\\Users\\feder\\Documents\\Deep Learning Project\\Data",True,["jpg"])
#FromImgToCsv("C:\\Users\\feder\\Documents\\Deep Learning Project\\Data", "C:\\Users\\feder\\Documents\\Deep Learning Project", True, 784, ["jpg"])
#print("END")
    