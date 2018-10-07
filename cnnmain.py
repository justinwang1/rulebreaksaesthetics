<<<<<<< HEAD
import os
import numpy as np
from PIL import Image
from util import *
import pickle
from convnet import *

def convDataFromNums(nums,num_map,size=16):
    data = []; new_nums = []
    for i in range(len(nums)):
        try:
            num = nums[i]
            id = num_map[num]
            img = Image.open('high/' + str(id) + '.jpg')
            img = img.resize((size, size))
            imgdata = PIL2array(img) / 255
            data.append(imgdata)
            new_nums.append(num)

        except ValueError:
            continue

    data = np.array(data); new_nums = np.array(new_nums)
    return [data,new_nums]

def getNegNums(pos_nums,all_nums,size=16):

    #Step 1: Figure out what the "negative" dataset numbers will be
    nums1 = pos_nums
    largest1 = np.max(nums1)
    idx1 = np.where(all_nums == largest1)[0][0]
    all_nums = all_nums[:idx1]

    positive_indices = np.nonzero(np.in1d(all_nums,nums1))[0]
    nums2 = np.delete(all_nums,positive_indices)
    if len(nums2) > len(nums1):
        nums2 = np.random.choice(nums2,size=len(nums1))

    return nums2

def prepFullData(data1,data2,nums=None,splitpts=[100,200],seed=123456):

    n1 = data1.shape[0]; n2 = data2.shape[0]; n = n1 + n2
    data = np.concatenate([data1,data2])

    #Prepare the labels
    lab1 = np.tile((0,1),(n1,1)); lab2 = np.tile((1,0),(n2,1))
    labels = np.concatenate([lab1,lab2])

    np.random.seed(seed)
    rdmz = np.random.permutation(n)
    data = data[rdmz,:,:,:]; labels = labels[rdmz,:]

    sp1,sp2 = splitpts
    indices_tr = np.arange(0,sp1); indices_val = np.arange(sp1,sp2); indices_ts = np.arange(sp2,n)
    data_tr = data[indices_tr,:,:,:]; labels_tr = labels[indices_tr,:]
    data_val = data[indices_val,:,:,:]; labels_val = labels[indices_val,:]
    data_ts = data[indices_ts,:,:,:]; labels_ts = labels[indices_ts,:]

    data_rlist = [data_tr,data_val,data_ts]; labels_rlist = [labels_tr,labels_val,labels_ts]
    overall_list = [data_rlist,labels_rlist]

    if nums is not None:
        nums = nums[rdmz]
        nums_tr = nums[indices_tr]; nums_val = nums[indices_val]; nums_ts = nums[indices_ts]
        nums_rlist = [nums_tr,nums_val,nums_ts]

        overall_list.append(nums_rlist)

    return overall_list

if __name__ == "__main__":

    #Subject to change for different tasks
    #filename = 'Silhouettes/blacksil.txt'
    #directory = 'Silhouettes/black/'

    filename = 'animals.txt'
    directory = 'Renumbered Data/high/'

    size = 32

    pos_nums = fileToNums(filename); all_nums = dirToNums(directory)
    neg_nums = getNegNums(pos_nums,all_nums)
    num_map = getRefDict()
    data1,nums1 = convDataFromNums(pos_nums,num_map,size=size)
    data2,nums2 = convDataFromNums(neg_nums,num_map,size=size)
    nums = np.concatenate([nums1,nums2])

    #data1 = pickle.load(open('Silhouettes/data_sil.p','rb'))
    #data2 = pickle.load(open('Silhouettes/data_nonsil.p','rb'))

    preppedData = prepFullData(data1,data2,nums=nums,splitpts=[200,350],seed=1234)
=======
import os
import numpy as np
from PIL import Image
from util import *
import pickle
from convnet import *

def convDataFromNums(nums,num_map,size=16):
    data = []; new_nums = []
    for i in range(len(nums)):
        try:
            num = nums[i]
            id = num_map[num]
            img = Image.open('high/' + str(id) + '.jpg')
            img = img.resize((size, size))
            imgdata = PIL2array(img) / 255
            data.append(imgdata)
            new_nums.append(num)

        except ValueError:
            continue

    data = np.array(data); new_nums = np.array(new_nums)
    return [data,new_nums]

def getNegNums(pos_nums,all_nums,size=16):

    #Step 1: Figure out what the "negative" dataset numbers will be
    nums1 = pos_nums
    largest1 = np.max(nums1)
    idx1 = np.where(all_nums == largest1)[0][0]
    all_nums = all_nums[:idx1]

    positive_indices = np.nonzero(np.in1d(all_nums,nums1))[0]
    nums2 = np.delete(all_nums,positive_indices)
    if len(nums2) > len(nums1):
        nums2 = np.random.choice(nums2,size=len(nums1))

    return nums2

def prepFullData(data1,data2,nums=None,splitpts=[100,200],seed=123456):

    n1 = data1.shape[0]; n2 = data2.shape[0]; n = n1 + n2
    data = np.concatenate([data1,data2])

    #Prepare the labels
    lab1 = np.tile((0,1),(n1,1)); lab2 = np.tile((1,0),(n2,1))
    labels = np.concatenate([lab1,lab2])

    np.random.seed(seed)
    rdmz = np.random.permutation(n)
    data = data[rdmz,:,:,:]; labels = labels[rdmz,:]

    sp1,sp2 = splitpts
    indices_tr = np.arange(0,sp1); indices_val = np.arange(sp1,sp2); indices_ts = np.arange(sp2,n)
    data_tr = data[indices_tr,:,:,:]; labels_tr = labels[indices_tr,:]
    data_val = data[indices_val,:,:,:]; labels_val = labels[indices_val,:]
    data_ts = data[indices_ts,:,:,:]; labels_ts = labels[indices_ts,:]

    data_rlist = [data_tr,data_val,data_ts]; labels_rlist = [labels_tr,labels_val,labels_ts]
    overall_list = [data_rlist,labels_rlist]

    if nums is not None:
        nums = nums[rdmz]
        nums_tr = nums[indices_tr]; nums_val = nums[indices_val]; nums_ts = nums[indices_ts]
        nums_rlist = [nums_tr,nums_val,nums_ts]

        overall_list.append(nums_rlist)

    return overall_list

if __name__ == "__main__":

    #Subject to change for different tasks
    #filename = 'Silhouettes/blacksil.txt'
    #directory = 'Silhouettes/black/'

    filename = 'animals.txt'
    directory = 'Renumbered Data/high/'

    size = 32

    pos_nums = fileToNums(filename); all_nums = dirToNums(directory)
    neg_nums = getNegNums(pos_nums,all_nums)
    num_map = getRefDict()
    data1,nums1 = convDataFromNums(pos_nums,num_map,size=size)
    data2,nums2 = convDataFromNums(neg_nums,num_map,size=size)
    nums = np.concatenate([nums1,nums2])

    #data1 = pickle.load(open('Silhouettes/data_sil.p','rb'))
    #data2 = pickle.load(open('Silhouettes/data_nonsil.p','rb'))

    preppedData = prepFullData(data1,data2,nums=nums,splitpts=[200,350],seed=1234)
>>>>>>> 8cc9dedaa868523276b391673922e48859692bb4
    model,error = train_model(preppedData,epochs=20,batch_size=8)