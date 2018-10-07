<<<<<<< HEAD
from PIL import Image,ImageDraw,ImageFont
import time
import os
import numpy as np
from matplotlib import colors as mcolors
from util import *
from colorhist import *


def colorSymmetry(img):

    color_list = getColorList(); color_names = getColorNames()
    X, Y = img.size
    imgMatHSV = PIL2array(img.convert('HSV')); imgMatHSV = imgMatHSV / float(255)

    # Initialize histograms
    colorhist = np.zeros((len(color_names)))
    windowsize = 16

    xmidpt = int(X/2)
    xrange = int(xmidpt/windowsize); yrange = int(Y/windowsize)

    total_similarity = 0; total_squares = yrange*xrange

    colorhist_left = np.zeros((len(color_names))); colorhist_right = np.zeros((len(color_names)))

    for i in range(yrange):
        ystart = i * windowsize; yend = min((i + 1) * windowsize, Y)
        for j in range(xrange):
            xstart_left = max(xmidpt - (j + 1)*windowsize,0)
            xend_left = xmidpt - j*windowsize

            xstart_right = xmidpt + j*windowsize
            xend_right = min(xmidpt + (j+1)*windowsize,X)

            currwindow_left = imgMatHSV[ystart:yend, xstart_left:xend_left, ]
            currwindow_right = imgMatHSV[ystart:yend, xstart_right:xend_right, ]

            pixelavg_left = np.apply_over_axes(np.mean, currwindow_left, [0, 1])[0][0]
            pixelavg_right = np.apply_over_axes(np.mean, currwindow_right, [0, 1])[0][0]

            w_left = whichColor(pixelavg_left, color_list)
            w_right = whichColor(pixelavg_right, color_list)
            similar = 1 if w_left == w_right else 0

            colorhist_left[w_left] += 1; colorhist_right[w_right] += 1

            #discrp = np.abs(w_left-w_right)
            total_similarity += similar
    sim_score = round(total_similarity/total_squares,4)
    colorhist_left = colorhist_left / np.sum(colorhist_left)
    colorhist_right = colorhist_right / np.sum(colorhist_right)
    #balance_score = round(np.linalg.norm(colorhist_left-colorhist_right),4)
    return sim_score


def objSymmetry(outline,windowsize=8):
    X,Y = outline.size
    xmidpt = int(X / 2)
    xrange = int(xmidpt / windowsize); yrange = int(Y / windowsize)
    total_discrp = 0; total_squares = yrange * xrange
    outline = PIL2array(outline)

    for i in range(yrange):
        ystart = i * windowsize; yend = min((i + 1) * windowsize, Y)
        for j in range(xrange):
            xstart_left = max(xmidpt - (j + 1) * windowsize, 0)
            xend_left = xmidpt - j * windowsize

            xstart_right = xmidpt + j * windowsize
            xend_right = min(xmidpt + (j + 1) * windowsize, X)

            windowavg_left = np.mean(outline[ystart:yend, xstart_left:xend_left, ])
            windowavg_right = np.mean(outline[ystart:yend, xstart_right:xend_right, ])
            discrp = np.abs(windowavg_left-windowavg_right)
            total_discrp += discrp
    print(total_discrp)

def main():
    os.chdir('C:/Users/jstwa/Desktop/ava/')
    num = 3218
    filename = 'Renumbered Data/high/' + str(num) + ".jpg"
    outlinename = 'Outlines/highrbd/' + str(num) + '.jpg'
    img = Image.open(filename); outline = Image.open(outlinename)
    test = objSymmetry(outline,windowsize=8)
    print(test)

if __name__ == '__main__':
    main()
=======
from PIL import Image,ImageDraw,ImageFont
import time
import os
import numpy as np
from matplotlib import colors as mcolors
from util import *
from colorhist import *


def colorSymmetry(img):

    color_list = getColorList(); color_names = getColorNames()
    X, Y = img.size
    imgMatHSV = PIL2array(img.convert('HSV')); imgMatHSV = imgMatHSV / float(255)

    # Initialize histograms
    colorhist = np.zeros((len(color_names)))
    windowsize = 16

    xmidpt = int(X/2)
    xrange = int(xmidpt/windowsize); yrange = int(Y/windowsize)

    total_similarity = 0; total_squares = yrange*xrange

    colorhist_left = np.zeros((len(color_names))); colorhist_right = np.zeros((len(color_names)))

    for i in range(yrange):
        ystart = i * windowsize; yend = min((i + 1) * windowsize, Y)
        for j in range(xrange):
            xstart_left = max(xmidpt - (j + 1)*windowsize,0)
            xend_left = xmidpt - j*windowsize

            xstart_right = xmidpt + j*windowsize
            xend_right = min(xmidpt + (j+1)*windowsize,X)

            currwindow_left = imgMatHSV[ystart:yend, xstart_left:xend_left, ]
            currwindow_right = imgMatHSV[ystart:yend, xstart_right:xend_right, ]

            pixelavg_left = np.apply_over_axes(np.mean, currwindow_left, [0, 1])[0][0]
            pixelavg_right = np.apply_over_axes(np.mean, currwindow_right, [0, 1])[0][0]

            w_left = whichColor(pixelavg_left, color_list)
            w_right = whichColor(pixelavg_right, color_list)
            similar = 1 if w_left == w_right else 0

            colorhist_left[w_left] += 1; colorhist_right[w_right] += 1

            #discrp = np.abs(w_left-w_right)
            total_similarity += similar
    sim_score = round(total_similarity/total_squares,4)
    colorhist_left = colorhist_left / np.sum(colorhist_left)
    colorhist_right = colorhist_right / np.sum(colorhist_right)
    #balance_score = round(np.linalg.norm(colorhist_left-colorhist_right),4)
    return sim_score


def objSymmetry(outline,windowsize=8):
    X,Y = outline.size
    xmidpt = int(X / 2)
    xrange = int(xmidpt / windowsize); yrange = int(Y / windowsize)
    total_discrp = 0; total_squares = yrange * xrange
    outline = PIL2array(outline)

    for i in range(yrange):
        ystart = i * windowsize; yend = min((i + 1) * windowsize, Y)
        for j in range(xrange):
            xstart_left = max(xmidpt - (j + 1) * windowsize, 0)
            xend_left = xmidpt - j * windowsize

            xstart_right = xmidpt + j * windowsize
            xend_right = min(xmidpt + (j + 1) * windowsize, X)

            windowavg_left = np.mean(outline[ystart:yend, xstart_left:xend_left, ])
            windowavg_right = np.mean(outline[ystart:yend, xstart_right:xend_right, ])
            discrp = np.abs(windowavg_left-windowavg_right)
            total_discrp += discrp
    print(total_discrp)

def main():
    os.chdir('C:/Users/jstwa/Desktop/ava/')
    num = 3218
    filename = 'Renumbered Data/high/' + str(num) + ".jpg"
    outlinename = 'Outlines/highrbd/' + str(num) + '.jpg'
    img = Image.open(filename); outline = Image.open(outlinename)
    test = objSymmetry(outline,windowsize=8)
    print(test)

if __name__ == '__main__':
    main()
>>>>>>> 8cc9dedaa868523276b391673922e48859692bb4
