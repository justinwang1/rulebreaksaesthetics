<<<<<<< HEAD
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from util import *
from PIL import Image
import numpy as np

#Convert a PyPlot Figure into a Pillow object
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

#Converts a histogram into a Pillow object for display.
def vizHist(colorhist,colornames,labels=False,threshline=0):
    fig, ax = plt.subplots()
    L = len(colornames)
    interval = max(L // 14,1)
    xlabs = np.array(range(L)) + 1
    ax.set_xticks(np.arange(0,L,interval))
    ax.bar(xlabs,colorhist, color=colornames)

    if labels:
        thresh = np.max(colorhist) * 0.05
        for i in range(L):
            h = xlabs[i]; v = colorhist[i]
            if v > thresh:
                ax.text(h - 1.5, v + 0.0005, str(h), color='red', fontsize=6)

    if threshline > 0: ax.axhline(y=threshline)

    fig.suptitle('Histogram')
    colors = FigureToPIL(fig)

    return colors

#Hist is a color histogram consisting of ~140 standard colors. Lifted from matplotlib. Shorthist consists of only
#~16 colors, combines many bins from hist to only display standard colors. This function, based on the above vizHist()
#merges the color histogram, short color histogram, and image itself into a combined photograph for display purposes.
def histImager(num,type='high',threshline=0,asCombined=True):
    img = Image.open('Renumbered Data/' + type + '/' + str(num) + '.jpg')
    hist = np.loadtxt('Histograms/' + type + '/' + str(num) + '.txt')
    color_names = getColorNames()
    short_hist,short_color_names = createShortHist(hist)
    histImg = vizHist(hist,color_names,labels=True)
    shortHistImg = vizHist(short_hist,short_color_names,threshline=threshline)

    all = [histImg,shortHistImg,img]
    if asCombined: all = merge3(all)

    return all

#Create a color dictionary based on matplotlib's color library.
def getColorDict():
    avafolder = "C:/Users/jstwa/Desktop/ava/"
    colordict = open(avafolder + "basiccolordict1.txt", 'r').readlines()
    colordict = np.array([x.rstrip() for x in colordict])
    lbreaks = np.concatenate([[-1], np.where(colordict == '')[0], [len(colordict)]])
    ncolors = len(lbreaks) - 1

    cd = {}; col_names = []
    for i in range(ncolors):
        start, end = lbreaks[i:i + 2]
        color_family = colordict[start + 1]; col_names.append(color_family)
        color_list = [x.split(': ')[1] for x in colordict[start + 2:end]]
        cd[color_family] = color_list
    return(cd)

#Grab a list of colors from matplotlib's color library.
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
    return color_list

#Similar to above, but only grab the names.
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

#Returns the name of the closest matching color to a particular pixel.
def whichColor(pixel, color_list):
    if pixel[2] <= 0.2:
        return 0
    else:
        color_tuples = np.array([x[0] for x in color_list])
        def dist(y): return np.linalg.norm(pixel - y)
        test = np.apply_along_axis(dist, 1, color_tuples)
        whichcolor = np.where(test == min(test))[0][0]
        return whichcolor

#Create the color histogram of a photograph. Uses a sliding window average with default window size 16. Speeds up the
#algorithm considerably. Empirical results indicate minimal loss of accuracy using window size 16 vs. window size 1.
def makeColorHist(img,windowsize=16):
    color_list = getColorList(); color_names = getColorNames()
    X,Y = img.size

    try:
        imgMatHSV = PIL2array(img.convert('HSV'))
    except ValueError:
        imgMatV = PIL2Darray(img)
        imgMatH = np.zeros_like(imgMatV); imgMatS = np.zeros_like(imgMatV)
        imgMatHSV = np.dstack([imgMatH, imgMatS, imgMatV])

    imgMatHSV = imgMatHSV/float(255)

    #Initialize histograms
    colorhist = np.zeros((len(color_names)))
    yrange = int(Y/windowsize); xrange = int(X/windowsize)

    #Create histograms by going through pixel widow averages of size [windowsize]
    for i in range(yrange):
        ystart = i * windowsize; yend = min((i+1)*windowsize,Y)
        for j in range(xrange):
            xstart = j*windowsize; xend = min((j+1)*windowsize,X)
            currwindow = imgMatHSV[ystart:yend,xstart:xend,]
            pixelavg = np.apply_over_axes(np.mean,currwindow,[0,1])[0][0]
            w = whichColor(pixelavg,color_list)
            colorhist[w] += 1

    colorhist = colorhist/np.sum(colorhist)
    return colorhist

#Create a short color histogram from the regular color histogram. The short color histogram has only ~16 colors.
def createShortHist(hist):
    colorDict = getColorDict()
    color_list = getColorList(); color_names = getColorNames()
    color_names = np.array(color_names)
    short_color_names = np.array(list(colorDict.keys()))
    short_hist = np.zeros(len(short_color_names))

    for key, value in colorDict.items():
        family_names = np.array(value)
        indices = np.nonzero(family_names[:, None] == color_names)[1]
        short_index = np.argwhere(short_color_names == key)
        short_hist[short_index] = np.sum(hist[indices])

=======
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from util import *
from PIL import Image
import numpy as np

#Convert a PyPlot Figure into a Pillow object
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

#Converts a histogram into a Pillow object for display.
def vizHist(colorhist,colornames,labels=False,threshline=0):
    fig, ax = plt.subplots()
    L = len(colornames)
    interval = max(L // 14,1)
    xlabs = np.array(range(L)) + 1
    ax.set_xticks(np.arange(0,L,interval))
    ax.bar(xlabs,colorhist, color=colornames)

    if labels:
        thresh = np.max(colorhist) * 0.05
        for i in range(L):
            h = xlabs[i]; v = colorhist[i]
            if v > thresh:
                ax.text(h - 1.5, v + 0.0005, str(h), color='red', fontsize=6)

    if threshline > 0: ax.axhline(y=threshline)

    fig.suptitle('Histogram')
    colors = FigureToPIL(fig)

    return colors

#Hist is a color histogram consisting of ~140 standard colors. Lifted from matplotlib. Shorthist consists of only
#~16 colors, combines many bins from hist to only display standard colors. This function, based on the above vizHist()
#merges the color histogram, short color histogram, and image itself into a combined photograph for display purposes.
def histImager(num,type='high',threshline=0,asCombined=True):
    img = Image.open('Renumbered Data/' + type + '/' + str(num) + '.jpg')
    hist = np.loadtxt('Histograms/' + type + '/' + str(num) + '.txt')
    color_names = getColorNames()
    short_hist,short_color_names = createShortHist(hist)
    histImg = vizHist(hist,color_names,labels=True)
    shortHistImg = vizHist(short_hist,short_color_names,threshline=threshline)

    all = [histImg,shortHistImg,img]
    if asCombined: all = merge3(all)

    return all

#Create a color dictionary based on matplotlib's color library.
def getColorDict():
    avafolder = "C:/Users/jstwa/Desktop/ava/"
    colordict = open(avafolder + "basiccolordict1.txt", 'r').readlines()
    colordict = np.array([x.rstrip() for x in colordict])
    lbreaks = np.concatenate([[-1], np.where(colordict == '')[0], [len(colordict)]])
    ncolors = len(lbreaks) - 1

    cd = {}; col_names = []
    for i in range(ncolors):
        start, end = lbreaks[i:i + 2]
        color_family = colordict[start + 1]; col_names.append(color_family)
        color_list = [x.split(': ')[1] for x in colordict[start + 2:end]]
        cd[color_family] = color_list
    return(cd)

#Grab a list of colors from matplotlib's color library.
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
    return color_list

#Similar to above, but only grab the names.
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

#Returns the name of the closest matching color to a particular pixel.
def whichColor(pixel, color_list):
    if pixel[2] <= 0.2:
        return 0
    else:
        color_tuples = np.array([x[0] for x in color_list])
        def dist(y): return np.linalg.norm(pixel - y)
        test = np.apply_along_axis(dist, 1, color_tuples)
        whichcolor = np.where(test == min(test))[0][0]
        return whichcolor

#Create the color histogram of a photograph. Uses a sliding window average with default window size 16. Speeds up the
#algorithm considerably. Empirical results indicate minimal loss of accuracy using window size 16 vs. window size 1.
def makeColorHist(img,windowsize=16):
    color_list = getColorList(); color_names = getColorNames()
    X,Y = img.size

    try:
        imgMatHSV = PIL2array(img.convert('HSV'))
    except ValueError:
        imgMatV = PIL2Darray(img)
        imgMatH = np.zeros_like(imgMatV); imgMatS = np.zeros_like(imgMatV)
        imgMatHSV = np.dstack([imgMatH, imgMatS, imgMatV])

    imgMatHSV = imgMatHSV/float(255)

    #Initialize histograms
    colorhist = np.zeros((len(color_names)))
    yrange = int(Y/windowsize); xrange = int(X/windowsize)

    #Create histograms by going through pixel widow averages of size [windowsize]
    for i in range(yrange):
        ystart = i * windowsize; yend = min((i+1)*windowsize,Y)
        for j in range(xrange):
            xstart = j*windowsize; xend = min((j+1)*windowsize,X)
            currwindow = imgMatHSV[ystart:yend,xstart:xend,]
            pixelavg = np.apply_over_axes(np.mean,currwindow,[0,1])[0][0]
            w = whichColor(pixelavg,color_list)
            colorhist[w] += 1

    colorhist = colorhist/np.sum(colorhist)
    return colorhist

#Create a short color histogram from the regular color histogram. The short color histogram has only ~16 colors.
def createShortHist(hist):
    colorDict = getColorDict()
    color_list = getColorList(); color_names = getColorNames()
    color_names = np.array(color_names)
    short_color_names = np.array(list(colorDict.keys()))
    short_hist = np.zeros(len(short_color_names))

    for key, value in colorDict.items():
        family_names = np.array(value)
        indices = np.nonzero(family_names[:, None] == color_names)[1]
        short_index = np.argwhere(short_color_names == key)
        short_hist[short_index] = np.sum(hist[indices])

>>>>>>> 8cc9dedaa868523276b391673922e48859692bb4
    return [short_hist,short_color_names]