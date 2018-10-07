<<<<<<< HEAD
import numpy as np

def getHistBundle():
    hhists = np.loadtxt('Histograms/high.txt'); lhists = np.loadtxt('Histograms/low.txt')
    hids = np.loadtxt('Histograms/highids.txt'); lids = np.loadtxt('Histograms/lowids.txt')
    hists = np.concatenate([hhists, lhists])
    ids = np.concatenate([hids, lids])
    labels = np.concatenate([np.ones(hhists.shape[0]), np.zeros(lhists.shape[0])])

    histbundle = [hists, ids, labels]
    return histbundle


def nnhist(imgnum, imgcat, histbundle, K=20):
    hists, ids, labels = histbundle
    catnum = 1 if imgcat == 'high' else 0
    index = np.where( np.logical_and(ids == imgnum, labels == catnum ) )[0][0]
    curr = hists[index]

    dists = np.sum(np.abs(curr - hists), axis=1)
    neighborinds = np.argpartition(dists, K + 1)[:(K + 1)]
    tot_sum = sum(labels[neighborinds])
    if imgcat == 'high': tot_sum = tot_sum - 1

    nn = tot_sum / K

    neighbors = ids[neighborinds]; labors = labels[neighborinds]

    try:
        delidx = np.where(neighbors == ids[index])[0][0]
    except IndexError:
        delidx = 0
    neighbors = np.delete(neighbors,delidx); labors = np.delete(labors,delidx)
    nids = np.transpose(np.vstack((neighbors,labors)))

=======
import numpy as np

def getHistBundle():
    hhists = np.loadtxt('Histograms/high.txt'); lhists = np.loadtxt('Histograms/low.txt')
    hids = np.loadtxt('Histograms/highids.txt'); lids = np.loadtxt('Histograms/lowids.txt')
    hists = np.concatenate([hhists, lhists])
    ids = np.concatenate([hids, lids])
    labels = np.concatenate([np.ones(hhists.shape[0]), np.zeros(lhists.shape[0])])

    histbundle = [hists, ids, labels]
    return histbundle


def nnhist(imgnum, imgcat, histbundle, K=20):
    hists, ids, labels = histbundle
    catnum = 1 if imgcat == 'high' else 0
    index = np.where( np.logical_and(ids == imgnum, labels == catnum ) )[0][0]
    curr = hists[index]

    dists = np.sum(np.abs(curr - hists), axis=1)
    neighborinds = np.argpartition(dists, K + 1)[:(K + 1)]
    tot_sum = sum(labels[neighborinds])
    if imgcat == 'high': tot_sum = tot_sum - 1

    nn = tot_sum / K

    neighbors = ids[neighborinds]; labors = labels[neighborinds]

    try:
        delidx = np.where(neighbors == ids[index])[0][0]
    except IndexError:
        delidx = 0
    neighbors = np.delete(neighbors,delidx); labors = np.delete(labors,delidx)
    nids = np.transpose(np.vstack((neighbors,labors)))

>>>>>>> 8cc9dedaa868523276b391673922e48859692bb4
    return [nn, nids]