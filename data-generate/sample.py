import json
import random


large = 13
middle = 13
small = 14
filename = 'result.json'
resname = 'label_CWE.json'
with open(filename,'r') as f:
    dict_CWE = json.load(f)
    large_arr = []
    middle_arr = []
    small_arr = []
    for k,v in dict_CWE.items():
        if v > 400:
            if k != 'no_relative':
                large_arr.append(k)
        elif (400 >= v )and(v > 80):
            middle_arr.append(k)
        elif 80 >= v :
            small_arr.append(k)
    if len(large_arr) >= large:
        num_large = large
    else:
        num_large = len(large_arr)    
    if len(middle_arr) >= middle:
        num_middle = middle
    else:
        num_middle = len(middle_arr)    
    if len(small_arr) >= small:
        num_small = small
    else:
        num_small = len(small_arr)    
    
    range_l = random.sample(range(0,len(large_arr)-1),num_large)
    range_m = random.sample(range(0,len(middle_arr)-1),num_middle)
    range_s = random.sample(range(0,len(small_arr)-1),num_small)
    
    list = 0
    res = {}
    for i in range(0,len(large_arr)-1):
        if i in range_l:
            res[list] = large_arr[i]
            list+=1
    for i in range(0,len(middle_arr)-1):
        if i in range_m:
            res[list] = middle_arr[i]
            list+=1
    for i in range(0,len(small_arr)-1):
        if i in range_s:
            res[list] = small_arr[i]
            list+=1
    with open(resname,'w') as w:
        json.dump(res,w)
    