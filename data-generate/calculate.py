import json
import os


path = '/small-ase2022/muVulDeePecker-master/source_files/upload_source_1/'
dict_CWE  = {}
num_dir = 0
def get_file(root_path):
    global dict_CWE,num_dir
    flag = 0
    isfile = 0
    name_CWE = ''
    #获取该目录下所有的文件名称和目录名称
    dir_or_files = os.listdir(root_path)
    for dir_file in dir_or_files:
        dir_file_path = os.path.join(root_path,dir_file)
        if os.path.isdir(dir_file_path):
            get_file(dir_file_path)
        else :
            isfile = 1
            if dir_file.find('CWE') != -1 :
                flag = 1 
                list_arr= dir_file.split('_')
                for name in list_arr:
                    if name.find('CWE') != -1:
                        name_CWE = name
                        break
    if isfile == 1:
        if flag == 1:
            if dict_CWE.__contains__(name_CWE):
                dict_CWE[name_CWE] = dict_CWE[name_CWE] + 1
            else:
                dict_CWE[name_CWE] = 1
            num_dir += 1
            print(str(num_dir))
            print(name_CWE+" + 1")
        else :
            if dict_CWE.__contains__('no_relative'):
                dict_CWE['no_relative'] = dict_CWE['no_relative'] + 1
            else:
                dict_CWE['no_relative'] = 1
            num_dir +=1
            print(str(num_dir))
            print("no relative + 1")
        
get_file(path)
with open('./result.json','w') as f:
    json.dump(dict_CWE, f)
                