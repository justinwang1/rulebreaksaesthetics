<<<<<<< HEAD
from PIL import Image
import numpy as np
import time
from util import *
from nncolorfeature import *
from ruleofthirdsfeature import *
from colorhist import *

##Aesthetic analysis of rule breaks in photographs. This is the main file for feature creation.

#Primary function for creating features from a photograph. The features are used (1) for classifying High Quality
#and Low Quality photographs, (2) Detection and measurement of violations of photographic rules. The former serves
#as a verification for the latter, which is the primary goal.
def getFeatures(num,cat,histbundle):
    img = Image.open('Photographs/' + cat + '/' + str(num) + '.jpg')
    outline,bin,select = selectOutline(num,cat)
    imgMatHSV,imgMatV = toHSVMats(img)
    hist = np.loadtxt('Histograms/' + cat + '/' + str(num) + '.txt')

    if select == 2: return [None,None]

    parea = percentarea(bin)
    lcontrast = lightcontrast(imgMatV,bin)
    size,aspect = sizeaspect(imgMatV)
    blurKe = blur(imgMatV)
    hcount = huecount(imgMatHSV)
    avgS,avgV = avgHSVs(imgMatHSV)
    nnscore,neighbors = nnhist(num,cat,histbundle)
    outline1,infos = newThirds(outline, 20, ac=1.25, a1=0.5, a2=0.8)
    csal,maxpsal,psal,thirds,reason = infos
    bigcc,cc = colorfulness(hist)

    full_list = np.array([parea,lcontrast,size,aspect,blurKe,hcount,avgS,avgV,nnscore,reason,bigcc,cc])
    return [full_list,neighbors]


#Features based on color histogram
def colorfulness(hist):
    shorthist,shortcolnames = createShortHist(hist)
    shortcounthist = shorthist[5:]
    bigthreshold = 0.02
    threshold = 0.01
    bigcolorfulness = len(np.where(hist > bigthreshold)[0])
    colorfulness = len(np.where(shortcounthist > threshold)[0])

    return [bigcolorfulness,colorfulness]

#Features based on outline (subject detection)
def percentarea(bin,threshold=0.7):
    perarea = np.count_nonzero(bin) / float(np.size(bin))
    if perarea >= threshold: return -1.0
    return perarea

def lightcontrast(imgMatV,bin):
    percent = percentarea(bin)
    if percent >= 0.7: return -1.0
    Bs = np.mean(imgMatV[np.where(bin == 255)]); Bb = np.mean(imgMatV[np.where(bin == 0)])
    #lightcontrast = np.absolute(np.log(Bs / Bb))
    lightcontrast = Bs/Bb
    return lightcontrast

#Basic features
def sizeaspect(imgMatV):
    Y,X = imgMatV.shape
    size = float(X)*Y; aspect = float(X)/Y
    return [size,aspect]

def blur(imgMatV):
    fast = np.ravel(np.absolute(np.fft.fft2(imgMatV)))
    blurke = np.size(np.where(fast > 5.0)) / float(np.size(fast))
    return blurke

def huecount(imgMatHSV):
    if (len(imgMatHSV.shape) < 3):
        hue_count = 1
    elif isgreyscale(imgMatHSV):
        hue_count = 1
    else:
        imgMatH = imgMatHSV[:,:,0]; imgMatS = imgMatHSV[:,:,1]; imgMatV = imgMatHSV[:,:,2]
        imgMatHr = np.ravel(imgMatH); imgMatSr = np.ravel(imgMatS); imgMatVr = np.ravel(imgMatV)
        hue_values = imgMatHr[np.logical_and(0.15 < imgMatVr, imgMatVr < 0.95,imgMatSr > 0.2)]
        histy = np.histogram(hue_values,bins=20)[0]
        m = np.max(histy); alpha = 0.05
        hue_count = np.sum(histy > m * alpha)
    return hue_count

def avgHSVs(imgMatHSV):
    if(len(imgMatHSV.shape) < 3):
        avgS = 0; avgV = np.mean(imgMatHSV)
    else:
        avgS = np.mean(imgMatHSV[:,:,1])
        avgV = np.mean(imgMatHSV[:,:,2])
    return [avgS,avgV]

def main():
    highnums = [97,169,366]; lownums = [19,63,271]
    histbundle = getHistBundle()

    cats = ['high','low']

    ##Make sure folders get created even when deleted

    #mkdir() checks if directory exists and creates it if not. Contained in util.py.
    mkdir('Features Data/high/'); mkdir('Features Data/low/')

    #For text files storing the K nearest neighbors (color-wise) for each photograph
    mkdir('Features Data/highhistnbhd/'); mkdir('Features Data/lowhistnbhd/')

    #Iterate through both High Quality (top 5% scoring of all photographs) and Low Quality (bottom 5%) datasets
    for cat in cats:
        nums = highnums if cat == 'high' else lownums

        #Iterate through each photograph in that dataset
        for i in range(len(nums)):
            try:
                num = nums[i]
                features,neighborids = getFeatures(num,cat,histbundle)

                #Each photograph is saved as a separate text file. This makes life a lot easier if program crashes,
                #etc. The individual text files are combined into a data frame prior to the analysis.
                if features is not None:
                    np.savetxt('Features Data/' + cat + '/' + str(num) + '.txt',features)
                    np.savetxt('Features Data/' + cat + 'histnbhd/' + str(num) + '.txt',neighborids)
                    print('Completed ' + cat + ' iteration ' + str(i))

            #In case there is a problem reading in the photograph or it doesn't exist
            except (IOError,IndexError) as e:
                print('Error at iteration ' + str(i))

if __name__ == "__main__":
=======
from PIL import Image
import numpy as np
import time
from util import *
from nncolorfeature import *
from ruleofthirdsfeature import *
from colorhist import *

##Aesthetic analysis of rule breaks in photographs. This is the main file for feature creation.

#Primary function for creating features from a photograph. The features are used (1) for classifying High Quality
#and Low Quality photographs, (2) Detection and measurement of violations of photographic rules. The former serves
#as a verification for the latter, which is the primary goal.
def getFeatures(num,cat,histbundle):
    img = Image.open('Photographs/' + cat + '/' + str(num) + '.jpg')
    outline,bin,select = selectOutline(num,cat)
    imgMatHSV,imgMatV = toHSVMats(img)
    hist = np.loadtxt('Histograms/' + cat + '/' + str(num) + '.txt')

    if select == 2: return [None,None]

    parea = percentarea(bin)
    lcontrast = lightcontrast(imgMatV,bin)
    size,aspect = sizeaspect(imgMatV)
    blurKe = blur(imgMatV)
    hcount = huecount(imgMatHSV)
    avgS,avgV = avgHSVs(imgMatHSV)
    nnscore,neighbors = nnhist(num,cat,histbundle)
    outline1,infos = newThirds(outline, 20, ac=1.25, a1=0.5, a2=0.8)
    csal,maxpsal,psal,thirds,reason = infos
    bigcc,cc = colorfulness(hist)

    full_list = np.array([parea,lcontrast,size,aspect,blurKe,hcount,avgS,avgV,nnscore,reason,bigcc,cc])
    return [full_list,neighbors]


#Features based on color histogram
def colorfulness(hist):
    shorthist,shortcolnames = createShortHist(hist)
    shortcounthist = shorthist[5:]
    bigthreshold = 0.02
    threshold = 0.01
    bigcolorfulness = len(np.where(hist > bigthreshold)[0])
    colorfulness = len(np.where(shortcounthist > threshold)[0])

    return [bigcolorfulness,colorfulness]

#Features based on outline (subject detection)
def percentarea(bin,threshold=0.7):
    perarea = np.count_nonzero(bin) / float(np.size(bin))
    if perarea >= threshold: return -1.0
    return perarea

def lightcontrast(imgMatV,bin):
    percent = percentarea(bin)
    if percent >= 0.7: return -1.0
    Bs = np.mean(imgMatV[np.where(bin == 255)]); Bb = np.mean(imgMatV[np.where(bin == 0)])
    #lightcontrast = np.absolute(np.log(Bs / Bb))
    lightcontrast = Bs/Bb
    return lightcontrast

#Basic features
def sizeaspect(imgMatV):
    Y,X = imgMatV.shape
    size = float(X)*Y; aspect = float(X)/Y
    return [size,aspect]

def blur(imgMatV):
    fast = np.ravel(np.absolute(np.fft.fft2(imgMatV)))
    blurke = np.size(np.where(fast > 5.0)) / float(np.size(fast))
    return blurke

def huecount(imgMatHSV):
    if (len(imgMatHSV.shape) < 3):
        hue_count = 1
    elif isgreyscale(imgMatHSV):
        hue_count = 1
    else:
        imgMatH = imgMatHSV[:,:,0]; imgMatS = imgMatHSV[:,:,1]; imgMatV = imgMatHSV[:,:,2]
        imgMatHr = np.ravel(imgMatH); imgMatSr = np.ravel(imgMatS); imgMatVr = np.ravel(imgMatV)
        hue_values = imgMatHr[np.logical_and(0.15 < imgMatVr, imgMatVr < 0.95,imgMatSr > 0.2)]
        histy = np.histogram(hue_values,bins=20)[0]
        m = np.max(histy); alpha = 0.05
        hue_count = np.sum(histy > m * alpha)
    return hue_count

def avgHSVs(imgMatHSV):
    if(len(imgMatHSV.shape) < 3):
        avgS = 0; avgV = np.mean(imgMatHSV)
    else:
        avgS = np.mean(imgMatHSV[:,:,1])
        avgV = np.mean(imgMatHSV[:,:,2])
    return [avgS,avgV]

def main():
    highnums = [97,169,366]; lownums = [19,63,271]
    histbundle = getHistBundle()

    cats = ['high','low']

    ##Make sure folders get created even when deleted

    #mkdir() checks if directory exists and creates it if not. Contained in util.py.
    mkdir('Features Data/high/'); mkdir('Features Data/low/')

    #For text files storing the K nearest neighbors (color-wise) for each photograph
    mkdir('Features Data/highhistnbhd/'); mkdir('Features Data/lowhistnbhd/')

    #Iterate through both High Quality (top 5% scoring of all photographs) and Low Quality (bottom 5%) datasets
    for cat in cats:
        nums = highnums if cat == 'high' else lownums

        #Iterate through each photograph in that dataset
        for i in range(len(nums)):
            try:
                num = nums[i]
                features,neighborids = getFeatures(num,cat,histbundle)

                #Each photograph is saved as a separate text file. This makes life a lot easier if program crashes,
                #etc. The individual text files are combined into a data frame prior to the analysis.
                if features is not None:
                    np.savetxt('Features Data/' + cat + '/' + str(num) + '.txt',features)
                    np.savetxt('Features Data/' + cat + 'histnbhd/' + str(num) + '.txt',neighborids)
                    print('Completed ' + cat + ' iteration ' + str(i))

            #In case there is a problem reading in the photograph or it doesn't exist
            except (IOError,IndexError) as e:
                print('Error at iteration ' + str(i))

if __name__ == "__main__":
>>>>>>> 8cc9dedaa868523276b391673922e48859692bb4
    main()