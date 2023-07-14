import csv
import os


path = "/small-ase2022/data/Dataset-sard/Vul/"

lable_array = []
for filename in os.listdir(path):
    print(filename)
    name_array = filename.split('_')
    length = len(name_array)-2
    flag = 0
    lable_name = ""
    for i in range(0,length-1):
        if name_array[i] == '':
            if lable_name not in lable_array:
                lable_array.append(lable_name)
            lable_name = ""
        if name_array[i].find('CWE') != -1:
            if flag == 1 :
                lable_name = ""
                continue
            else :
                flag = 1
            continue
        if flag == 1:
            if lable_name == "":
                lable_name = name_array[i]
            else :
                lable_name=lable_name+'_'+name_array[i]
    if lable_name not in lable_array:
        lable_array.append(lable_name)

f = open("type.csv",'w')
csv_writer = csv.writer(f)
label_res = []
for i in lable_array:
    arr = i.split('_')
    if len(arr) >= 2:
        new_name = arr[0]+'_'+arr[1]
    else :
        new_name = arr[0]
    if new_name not in label_res:
        label_res.append(new_name)
    #csv_writer.writerow([i])
    #print(i)
for i in label_res:
    csv_writer.writerow([i])
    print(i)
length = len(label_res)
print(length)
f.close()