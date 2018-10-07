<<<<<<< HEAD
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import *
import os
from PIL import Image,ImageDraw,ImageFont
from matplotlib import colors as mcolors


def selectOutline(num,cat='high'):

    def percentarea(bin, threshold=0.7):
        perarea = np.count_nonzero(bin) / float(np.size(bin))
        if perarea >= threshold: return -1.0
        return perarea

    #0 = RBD, 1 = MBD (RBD missing) 2 = MBD (RBD % too high) 3 = Neither

    rbd_path = 'Outlines/' + cat + 'rbd/' + str(num) + '.jpg'
    mbd_path = 'Outlines/' + cat + '/' + str(num) + '.jpg'

    if os.path.isfile(rbd_path):
        outline = Image.open(rbd_path)
        outline,bin = cleanRBD(outline)
        per = percentarea(bin,threshold=0.75)
        if per == -1:
            try:
                outline = Image.open(mbd_path)
                bin = binMBD(outline)
                pkg = [outline,bin,2]
            except FileNotFoundError:
                pkg = [outline,bin,0]
        else:
            pkg = [outline,bin,0]

    elif os.path.isfile(mbd_path):
        outline = Image.open(mbd_path)
        bin = binMBD(outline)
        pkg = [outline,bin,1]
    else:
        pkg = [None,None,3]

    return pkg


def idlist(dir,savefile=''):
    ids = os.listdir(dir)
    ids = [int(x.split('.')[0]) for x in ids]
    ids = np.sort(ids)

    if savefile != '':
        with open(savefile,'w') as f:
            for x in ids:
                f.write(str(x) + '\n')
            f.close()

    return ids


#Make a directory if it doesn't exist already

def mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


#Change to AVA directory. Convenient shortcut.
def avadir(local=True):
    maindir = 'C:/Users/jstwa/Desktop/ava/' if local else '/home/justinwa/Aesth/'
    os.chdir(maindir)

#Convert PIL Image to Array
def toHSVMats(img):
    try:
        img = img.convert('HSV')
        imgMatHSV = PIL2array(img); imgMatV = imgMatHSV[:, :, 2]
    except ValueError:
        imgMatHSV = PIL2Darray(img); imgMatV = imgMatHSV
        print("No H and S matrices here.")

    imgMatHSV = imgMatHSV / 255; imgMatV = imgMatV / 255
    return [imgMatHSV,imgMatV]


def PIL2Darray(img):
    return np.array(img.getdata(),np.uint8).reshape(img.size[1], img.size[0])

def PIL2array(img):
    return np.array(img.getdata(),
                    np.uint8).reshape(img.size[1], img.size[0], 3)

def PIL2array3(img):
    return np.array(img.getdata(),
                    np.uint8).reshape(img.size[1], img.size[0], 3)


#Clean outlines
def cleanRBD(outline,thresh=100):
    outlineMat = PIL2array(outline); outlineMat[np.where(outlineMat <= thresh)] = 0
    outlinescrub = Image.fromarray(outlineMat)
    bin = outlineMat.copy(); bin[np.where(bin > 0)] = 255
    bin = np.mean(bin, axis=2)
    return [outlinescrub,bin]

def binMBD(outline,thresh=60):
    outlineMat = PIL2array(outline)
    bin = np.zeros_like(outlineMat); bin[np.where(outlineMat > thresh)] = 255
    bin = np.mean(bin, axis=2)
    return bin

#Misc.

def getRefDict(filepath='Renumbered Data/highref.txt'):

    ref = open(filepath, 'r').readlines()
    ref = [x.rstrip() for x in ref]

    num_map = {}
    for line in ref:
        sil_num, id_num = line.split('. ')
        sil_num = int(sil_num); id_num = int(id_num)
        num_map[sil_num] = id_num

    return num_map

def isgreyscale(imgMat):
    return np.count_nonzero(imgMat[:, :, 0]) == 0


def dirToNums(dir):
    nums = os.listdir(dir)
    nums = [int(x.rstrip().split('.')[0]) for x in nums]
    nums = np.sort(np.array(nums))
    return nums

def fileToNums(filename):
    nums = open(filename, 'r').readlines()
    nums = [int(x.rstrip()) for x in nums]
    nums = np.sort(np.array(nums))
    return nums


#Merging of Images
def merge2(images):
    w,h = images[0].size

    new_im = Image.new('RGB', (2*w, h))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    return new_im

def merge3(images):
    im0, im1, im2 = images
    widths = [im0.size[0], im1.size[0], im2.size[0]]; w = max(widths)
    heights = [im0.size[1], im1.size[1], im2.size[0]]; h = max(heights)
    new_im = Image.new('RGB', (2 * w, 2 * h))
    new_im.paste(im0,(0,0)); new_im.paste(im1,(w,0)); new_im.paste(im2,(int(w/2),h))
    return new_im

def merge4(images):
    im0,im1,im2,im3 = images
    widths = [im0.size[0], im1.size[0], im2.size[0], im3.size[0]]; w = max(widths)
    heights = [im0.size[1], im1.size[1], im2.size[0], im3.size[1]]; h = max(heights)
    new_im = Image.new('RGB', (2*w, 2*h))

    new_im.paste(im0,(0,0)); new_im.paste(im1,(w,0))
    new_im.paste(im2,(0,h)); new_im.paste(im3,(w,h))

    return new_im


#Histograms
def FigureToPIL(fig):
    def fig2data(fig):
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
        buf.shape = (w, h, 4)
        buf = np.roll(buf, 3, axis=2)
        return buf

    buf = fig2data(fig)
    w, h, d = buf.shape
    return Image.frombytes( "RGBA", ( w ,h ), buf.tostring( ) )


def getColorList():
    # Get color names and coordinates of all colors in Pyplot library
    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)
                    for name, color in colors.items())
    sorted_names = [name for hsv, name in by_hsv]

    repeat_colors = np.array(['k', 'dimgrey', 'grey', 'darkgrey', 'lightgrey', 'w', 'r', 'g', 'b'])
    color_list = []
    for i in range(len(by_hsv)):
        curr = by_hsv[i]; colour = curr[1]
        if colour not in repeat_colors: color_list.append(curr)
    color_names = np.array([(x[1]) for x in color_list])
    return [color_list,color_names]

def getColorNames():
    # Get color names and coordinates of all colors in Pyplot library
    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)
                    for name, color in colors.items())
    sorted_names = [name for hsv, name in by_hsv]

    repeat_colors = np.array(['k', 'dimgrey', 'grey', 'darkgrey', 'lightgrey', 'w', 'r', 'g', 'b'])
    color_list = []
    for i in range(len(by_hsv)):
        curr = by_hsv[i]; colour = curr[1]
        if colour not in repeat_colors: color_list.append(curr)
    color_names = np.array([(x[1]) for x in color_list])
    return color_names

#Plot a matrix
def plot(imgMat):
    plt.matshow(imgMat)
    plt.show()

#Modify an outline
def modoutline(outline):
    X, Y = outline.size
    width = 20
    Xc, Yc = int(float(X) / 2), int(float(Y) / 2)
    X1, Y1 = int(float(X) / 3), int(float(Y) / 3)
    X3, Y3 = int(2 * float(X) / 3), int(2 * float(Y) / 3)
    pairs = [(X1, Y1), (X3, Y1), (X1, Y3), (X3, Y3)]
    draw1 = ImageDraw.Draw(outline)
    draw1.rectangle([Xc - width, Yc - width, Xc + width, Yc + width], outline="white")
    width1 = 40
    for k in range(4):
        Xp, Yp = pairs[k]
        draw1.rectangle([Xp - width1, Yp - width1, Xp + width1, Yp + width1], outline="white")
    return outline

#Make Info Image
def makeInfoImg(img,names,vals,onecol=True):
    info = Image.new('RGB', img.size); draw = ImageDraw.Draw(info)
    X,Y = img.size
    fsize = int(Y / 20); offset = 1.8 * fsize
    font = ImageFont.truetype("arial.ttf", fsize)
    xplus = int(X / 2.0) if X < Y else int(X / 3.2)
    ypos = 30

    for k in range(len(names)):
        xmove = 0 if (k % 2 == 0 or onecol) else xplus
        draw.text((30 + xmove, ypos), names[k] + ": " + str(vals[k]), font=font)
        if (xmove == xplus or onecol): ypos += offset

=======
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import *
import os
from PIL import Image,ImageDraw,ImageFont
from matplotlib import colors as mcolors


def selectOutline(num,cat='high'):

    def percentarea(bin, threshold=0.7):
        perarea = np.count_nonzero(bin) / float(np.size(bin))
        if perarea >= threshold: return -1.0
        return perarea

    #0 = RBD, 1 = MBD (RBD missing) 2 = MBD (RBD % too high) 3 = Neither

    rbd_path = 'Outlines/' + cat + 'rbd/' + str(num) + '.jpg'
    mbd_path = 'Outlines/' + cat + '/' + str(num) + '.jpg'

    if os.path.isfile(rbd_path):
        outline = Image.open(rbd_path)
        outline,bin = cleanRBD(outline)
        per = percentarea(bin,threshold=0.75)
        if per == -1:
            try:
                outline = Image.open(mbd_path)
                bin = binMBD(outline)
                pkg = [outline,bin,2]
            except FileNotFoundError:
                pkg = [outline,bin,0]
        else:
            pkg = [outline,bin,0]

    elif os.path.isfile(mbd_path):
        outline = Image.open(mbd_path)
        bin = binMBD(outline)
        pkg = [outline,bin,1]
    else:
        pkg = [None,None,3]

    return pkg


def idlist(dir,savefile=''):
    ids = os.listdir(dir)
    ids = [int(x.split('.')[0]) for x in ids]
    ids = np.sort(ids)

    if savefile != '':
        with open(savefile,'w') as f:
            for x in ids:
                f.write(str(x) + '\n')
            f.close()

    return ids


#Make a directory if it doesn't exist already

def mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


#Change to AVA directory. Convenient shortcut.
def avadir(local=True):
    maindir = 'C:/Users/jstwa/Desktop/ava/' if local else '/home/justinwa/Aesth/'
    os.chdir(maindir)

#Convert PIL Image to Array
def toHSVMats(img):
    try:
        img = img.convert('HSV')
        imgMatHSV = PIL2array(img); imgMatV = imgMatHSV[:, :, 2]
    except ValueError:
        imgMatHSV = PIL2Darray(img); imgMatV = imgMatHSV
        print("No H and S matrices here.")

    imgMatHSV = imgMatHSV / 255; imgMatV = imgMatV / 255
    return [imgMatHSV,imgMatV]


def PIL2Darray(img):
    return np.array(img.getdata(),np.uint8).reshape(img.size[1], img.size[0])

def PIL2array(img):
    return np.array(img.getdata(),
                    np.uint8).reshape(img.size[1], img.size[0], 3)

def PIL2array3(img):
    return np.array(img.getdata(),
                    np.uint8).reshape(img.size[1], img.size[0], 3)


#Clean outlines
def cleanRBD(outline,thresh=100):
    outlineMat = PIL2array(outline); outlineMat[np.where(outlineMat <= thresh)] = 0
    outlinescrub = Image.fromarray(outlineMat)
    bin = outlineMat.copy(); bin[np.where(bin > 0)] = 255
    bin = np.mean(bin, axis=2)
    return [outlinescrub,bin]

def binMBD(outline,thresh=60):
    outlineMat = PIL2array(outline)
    bin = np.zeros_like(outlineMat); bin[np.where(outlineMat > thresh)] = 255
    bin = np.mean(bin, axis=2)
    return bin

#Misc.

def getRefDict(filepath='Renumbered Data/highref.txt'):

    ref = open(filepath, 'r').readlines()
    ref = [x.rstrip() for x in ref]

    num_map = {}
    for line in ref:
        sil_num, id_num = line.split('. ')
        sil_num = int(sil_num); id_num = int(id_num)
        num_map[sil_num] = id_num

    return num_map

def isgreyscale(imgMat):
    return np.count_nonzero(imgMat[:, :, 0]) == 0


def dirToNums(dir):
    nums = os.listdir(dir)
    nums = [int(x.rstrip().split('.')[0]) for x in nums]
    nums = np.sort(np.array(nums))
    return nums

def fileToNums(filename):
    nums = open(filename, 'r').readlines()
    nums = [int(x.rstrip()) for x in nums]
    nums = np.sort(np.array(nums))
    return nums


#Merging of Images
def merge2(images):
    w,h = images[0].size

    new_im = Image.new('RGB', (2*w, h))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    return new_im

def merge3(images):
    im0, im1, im2 = images
    widths = [im0.size[0], im1.size[0], im2.size[0]]; w = max(widths)
    heights = [im0.size[1], im1.size[1], im2.size[0]]; h = max(heights)
    new_im = Image.new('RGB', (2 * w, 2 * h))
    new_im.paste(im0,(0,0)); new_im.paste(im1,(w,0)); new_im.paste(im2,(int(w/2),h))
    return new_im

def merge4(images):
    im0,im1,im2,im3 = images
    widths = [im0.size[0], im1.size[0], im2.size[0], im3.size[0]]; w = max(widths)
    heights = [im0.size[1], im1.size[1], im2.size[0], im3.size[1]]; h = max(heights)
    new_im = Image.new('RGB', (2*w, 2*h))

    new_im.paste(im0,(0,0)); new_im.paste(im1,(w,0))
    new_im.paste(im2,(0,h)); new_im.paste(im3,(w,h))

    return new_im


#Histograms
def FigureToPIL(fig):
    def fig2data(fig):
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
        buf.shape = (w, h, 4)
        buf = np.roll(buf, 3, axis=2)
        return buf

    buf = fig2data(fig)
    w, h, d = buf.shape
    return Image.frombytes( "RGBA", ( w ,h ), buf.tostring( ) )


def getColorList():
    # Get color names and coordinates of all colors in Pyplot library
    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)
                    for name, color in colors.items())
    sorted_names = [name for hsv, name in by_hsv]

    repeat_colors = np.array(['k', 'dimgrey', 'grey', 'darkgrey', 'lightgrey', 'w', 'r', 'g', 'b'])
    color_list = []
    for i in range(len(by_hsv)):
        curr = by_hsv[i]; colour = curr[1]
        if colour not in repeat_colors: color_list.append(curr)
    color_names = np.array([(x[1]) for x in color_list])
    return [color_list,color_names]

def getColorNames():
    # Get color names and coordinates of all colors in Pyplot library
    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)
                    for name, color in colors.items())
    sorted_names = [name for hsv, name in by_hsv]

    repeat_colors = np.array(['k', 'dimgrey', 'grey', 'darkgrey', 'lightgrey', 'w', 'r', 'g', 'b'])
    color_list = []
    for i in range(len(by_hsv)):
        curr = by_hsv[i]; colour = curr[1]
        if colour not in repeat_colors: color_list.append(curr)
    color_names = np.array([(x[1]) for x in color_list])
    return color_names

#Plot a matrix
def plot(imgMat):
    plt.matshow(imgMat)
    plt.show()

#Modify an outline
def modoutline(outline):
    X, Y = outline.size
    width = 20
    Xc, Yc = int(float(X) / 2), int(float(Y) / 2)
    X1, Y1 = int(float(X) / 3), int(float(Y) / 3)
    X3, Y3 = int(2 * float(X) / 3), int(2 * float(Y) / 3)
    pairs = [(X1, Y1), (X3, Y1), (X1, Y3), (X3, Y3)]
    draw1 = ImageDraw.Draw(outline)
    draw1.rectangle([Xc - width, Yc - width, Xc + width, Yc + width], outline="white")
    width1 = 40
    for k in range(4):
        Xp, Yp = pairs[k]
        draw1.rectangle([Xp - width1, Yp - width1, Xp + width1, Yp + width1], outline="white")
    return outline

#Make Info Image
def makeInfoImg(img,names,vals,onecol=True):
    info = Image.new('RGB', img.size); draw = ImageDraw.Draw(info)
    X,Y = img.size
    fsize = int(Y / 20); offset = 1.8 * fsize
    font = ImageFont.truetype("arial.ttf", fsize)
    xplus = int(X / 2.0) if X < Y else int(X / 3.2)
    ypos = 30

    for k in range(len(names)):
        xmove = 0 if (k % 2 == 0 or onecol) else xplus
        draw.text((30 + xmove, ypos), names[k] + ": " + str(vals[k]), font=font)
        if (xmove == xplus or onecol): ypos += offset

>>>>>>> 8cc9dedaa868523276b391673922e48859692bb4
    return info