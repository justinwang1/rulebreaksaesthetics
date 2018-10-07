<<<<<<< HEAD
from util import *
import numpy as np
import os

def thirdsCriteria(csal,psal,ac,a1,a2):
    maxpsal = np.max(psal)
    c1,c2,c3,c4 = psal
    cthresh = np.round(csal * ac, 0)
    lthresh = np.round(np.max([csal, maxpsal]) * a1, 0)
    hthresh = np.round(np.max([csal, maxpsal]) * a2,0)
    thirds = 'no'; reason = 0

    #The most obvious: Center saliency is less than max corner saliency
    if maxpsal > cthresh:
        thirds = 'yes'; reason = 1

    #Off-center: Object on bottom
    if np.max([c1,c2]) < lthresh and np.min([c3,c4]) > hthresh:
        thirds = 'yes'; reason = 2

    #Off-center: Object on top
    elif np.max([c3,c4]) < lthresh and np.min([c1,c2]) > hthresh:
        thirds = 'yes'; reason = 3

    #Off-center: Object on left
    elif np.max([c1,c3]) < lthresh and np.min([c2,c4]) > hthresh:
        thirds = 'yes'; reason = 4

    #Off-center: Object on right
    elif np.max([c2,c4]) < lthresh and np.min([c1,c3]) > hthresh:
        thirds = 'yes'; reason = 5

    return [thirds,reason]


def newThirds(outline,boxwidth=20,ac=1.25,a1=0.5,a2=0.75):

    X,Y = outline.size
    outlineMat = PIL2array(outline)

    #Measure rule of thirds
    Xc,Yc = int(float(X)/2),int(float(Y)/2)
    cbox = outlineMat[(Yc - boxwidth):(Yc + boxwidth),(Xc - boxwidth):(Xc + boxwidth)]
    csal = round(np.mean(cbox),2)
    X1,Y1 = int(float(X)/3),int(float(Y)/3); X3,Y3 = int(2*float(X) / 3), int(2*float(Y) / 3)
    pairs = [(X1,Y1),(X3,Y1),(X1,Y3),(X3,Y3)]

    psal = []
    for i in range(4):
        Xp,Yp = pairs[i]
        abox = outlineMat[(Yp - boxwidth):(Yp + boxwidth),(Xp - boxwidth):(Xp + boxwidth)]
        asal = round(np.mean(abox),2)
        psal.append(asal)
    maxpsal = np.max(psal)

    thirds,reason = thirdsCriteria(csal,psal,ac,a1,a2)
    #off = offcenter(psal)

    #Original Image Modified
    outline1 = outline.copy()
    draw1 = ImageDraw.Draw(outline1)
    draw1.rectangle([Xc-boxwidth,Yc-boxwidth,Xc+boxwidth,Yc+boxwidth],outline="white")
    for i in range(4):
        Xp, Yp = pairs[i]
        draw1.rectangle([Xp - boxwidth, Yp - boxwidth, Xp + boxwidth, Yp + boxwidth],outline="white")

    return([outline1,[csal,maxpsal,psal,thirds,reason]])

def main():
    avadir()
    folder = 'high'
    nums = os.listdir('Outlines/' + folder + 'rbd/')
    nums = [x.rstrip().split('.')[0] for x in nums]
    thirdscount = 0

    niters = 2000
    for i in range(niters):

        #Setup
        num = nums[i]
        img = Image.open(folder + '/' + str(num) + '.jpg')
        outline = Image.open('Outlines/' + folder + 'rbd/' + str(num) + '.jpg')

        #Set parameters and run thirds algorithm
        ac = 1.25; a1 = 0.6; a2 = 0.8
        outline1,infos = newThirds(outline,20,ac,a1,a2)
        csal,maxpsal,psal,thirds,reason = infos

        #Skip if doesn't meet criteria
        if thirds == 'yes': continue

        #Recalcuating in-algorithm parameters for display purposes
        cthresh = np.round(csal * ac, 0)
        lthresh = np.round(np.max([csal, maxpsal]) * a1, 0)
        hthresh = np.round(np.max([csal, maxpsal]) * a2,0)

        #Creating the info image
        rdescs = ['N/A','Classic Corner','Off-Center - Bottom','Off-center - Top',
                  'Off-center - Left','Off-center - Right']
        rd = rdescs[reason]

        names = ['Center Sal','Sal 1','Sal 2','Sal 3','Sal 4','Thirds','C Thresh','Low Thresh','High Thresh','Reason']
        vals = [csal] + psal + [thirds,cthresh,lthresh,hthresh,rd]
        infos = makeInfoImg(img,names,vals)

        #Merge and save
        all = merge3([img,outline1,infos])
        all.save('RuleOfThirds/' + folder + 'rbdmiss/' + str(num) + '.jpg')

        print('Completed iteration ' + str(i))

if __name__ == "__main__":
=======
from util import *
import numpy as np
import os

def thirdsCriteria(csal,psal,ac,a1,a2):
    maxpsal = np.max(psal)
    c1,c2,c3,c4 = psal
    cthresh = np.round(csal * ac, 0)
    lthresh = np.round(np.max([csal, maxpsal]) * a1, 0)
    hthresh = np.round(np.max([csal, maxpsal]) * a2,0)
    thirds = 'no'; reason = 0

    #The most obvious: Center saliency is less than max corner saliency
    if maxpsal > cthresh:
        thirds = 'yes'; reason = 1

    #Off-center: Object on bottom
    if np.max([c1,c2]) < lthresh and np.min([c3,c4]) > hthresh:
        thirds = 'yes'; reason = 2

    #Off-center: Object on top
    elif np.max([c3,c4]) < lthresh and np.min([c1,c2]) > hthresh:
        thirds = 'yes'; reason = 3

    #Off-center: Object on left
    elif np.max([c1,c3]) < lthresh and np.min([c2,c4]) > hthresh:
        thirds = 'yes'; reason = 4

    #Off-center: Object on right
    elif np.max([c2,c4]) < lthresh and np.min([c1,c3]) > hthresh:
        thirds = 'yes'; reason = 5

    return [thirds,reason]


def newThirds(outline,boxwidth=20,ac=1.25,a1=0.5,a2=0.75):

    X,Y = outline.size
    outlineMat = PIL2array(outline)

    #Measure rule of thirds
    Xc,Yc = int(float(X)/2),int(float(Y)/2)
    cbox = outlineMat[(Yc - boxwidth):(Yc + boxwidth),(Xc - boxwidth):(Xc + boxwidth)]
    csal = round(np.mean(cbox),2)
    X1,Y1 = int(float(X)/3),int(float(Y)/3); X3,Y3 = int(2*float(X) / 3), int(2*float(Y) / 3)
    pairs = [(X1,Y1),(X3,Y1),(X1,Y3),(X3,Y3)]

    psal = []
    for i in range(4):
        Xp,Yp = pairs[i]
        abox = outlineMat[(Yp - boxwidth):(Yp + boxwidth),(Xp - boxwidth):(Xp + boxwidth)]
        asal = round(np.mean(abox),2)
        psal.append(asal)
    maxpsal = np.max(psal)

    thirds,reason = thirdsCriteria(csal,psal,ac,a1,a2)
    #off = offcenter(psal)

    #Original Image Modified
    outline1 = outline.copy()
    draw1 = ImageDraw.Draw(outline1)
    draw1.rectangle([Xc-boxwidth,Yc-boxwidth,Xc+boxwidth,Yc+boxwidth],outline="white")
    for i in range(4):
        Xp, Yp = pairs[i]
        draw1.rectangle([Xp - boxwidth, Yp - boxwidth, Xp + boxwidth, Yp + boxwidth],outline="white")

    return([outline1,[csal,maxpsal,psal,thirds,reason]])

def main():
    avadir()
    folder = 'high'
    nums = os.listdir('Outlines/' + folder + 'rbd/')
    nums = [x.rstrip().split('.')[0] for x in nums]
    thirdscount = 0

    niters = 2000
    for i in range(niters):

        #Setup
        num = nums[i]
        img = Image.open(folder + '/' + str(num) + '.jpg')
        outline = Image.open('Outlines/' + folder + 'rbd/' + str(num) + '.jpg')

        #Set parameters and run thirds algorithm
        ac = 1.25; a1 = 0.6; a2 = 0.8
        outline1,infos = newThirds(outline,20,ac,a1,a2)
        csal,maxpsal,psal,thirds,reason = infos

        #Skip if doesn't meet criteria
        if thirds == 'yes': continue

        #Recalcuating in-algorithm parameters for display purposes
        cthresh = np.round(csal * ac, 0)
        lthresh = np.round(np.max([csal, maxpsal]) * a1, 0)
        hthresh = np.round(np.max([csal, maxpsal]) * a2,0)

        #Creating the info image
        rdescs = ['N/A','Classic Corner','Off-Center - Bottom','Off-center - Top',
                  'Off-center - Left','Off-center - Right']
        rd = rdescs[reason]

        names = ['Center Sal','Sal 1','Sal 2','Sal 3','Sal 4','Thirds','C Thresh','Low Thresh','High Thresh','Reason']
        vals = [csal] + psal + [thirds,cthresh,lthresh,hthresh,rd]
        infos = makeInfoImg(img,names,vals)

        #Merge and save
        all = merge3([img,outline1,infos])
        all.save('RuleOfThirds/' + folder + 'rbdmiss/' + str(num) + '.jpg')

        print('Completed iteration ' + str(i))

if __name__ == "__main__":
>>>>>>> 8cc9dedaa868523276b391673922e48859692bb4
    main()