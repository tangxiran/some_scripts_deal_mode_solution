# encoding=utf-8
def save_npy_to_mat(file='40mode_no_duicheng_new_2020_12_30.npy', matFile='mat//mode_data'):
    import numpy as np
    file = file
    npy = np.load(file)
    import scipy.io as io
    io.savemat(matFile, {'mode_data_all': npy
                         })

# 特定的文件后缀保存，筛选后的特定后缀
def file_select(data_dir,*args):  #
    # data_dir is file_dir//
    # *args is 'png','jpeg','bmp' so on
    import  glob
    # *args是要筛选的的后缀名称可以选择多个
    length = len(args)
    file_list = []
    for i in range(length):
        file_list = file_list + list(glob.glob(data_dir + '/*.'+str(args[i])))
    # file_list = list(glob.glob(data_dir + '/*.png')) + list(glob.glob(data_dir + '/*.jpg'))   # get name list of all .png files
    # data = []
    # print(file_list) # 得到文件的路径列表
    return file_list


def find_start(data_list = [],sub_str='Ex'):
    Ex = sub_str
    for i in range(len(data_list)):
        if Ex in data_list[i]:
            # print('is line start occur:',i)
            start_line = i
            break
    return start_line

def save_txt_to_numpy(txt_file,numpy_file,mode_number=40,jingdu_prest= 50+1):
    import pandas as pd
    import numpy as np
    pd_result = pd.read_table(filepath_or_buffer=txt_file, sep=',')
    mode_40_origin = np.array(pd_result)
    print(mode_40_origin.shape)
    mode_jingdu = int(((mode_40_origin.shape)[0])**0.5)

    print('mode jingdu is ',mode_jingdu)
    data_result = np.zeros(shape=(mode_number,3, int(mode_jingdu ** 2)), dtype=complex)
    for i in range(mode_number):
        for k in range(int(mode_jingdu ** 2)):
            mode_start = i * 12
            ex_real_start = mode_start + 0
            ex_img_start = ex_real_start + 1

            ey_real_start = mode_start + 2
            ey_img_start = ey_real_start + 1

            ez_real_start = mode_start + 4
            ez_img_start = ez_real_start + 1

            data_result[i, 0, k] = mode_40_origin[k, ex_real_start] + mode_40_origin[k, ex_img_start] * 1.0j
            data_result[i, 1, k] = mode_40_origin[k, ey_real_start] + mode_40_origin[k, ey_img_start] * 1.0j
            data_result[i, 2, k] = mode_40_origin[k, ez_real_start] + mode_40_origin[k, ez_img_start] * 1.0j

    print('date_complex shape is ',data_result.shape)
    output = []
    for i in range(mode_number):
        mode_output= []
        for j in range(3):
            data = np.reshape(data_result[i,j,:],(mode_jingdu,mode_jingdu))
            data = data[mode_jingdu//2-jingdu_prest//2:mode_jingdu//2+jingdu_prest//2+1,
                   mode_jingdu//2-jingdu_prest//2:mode_jingdu//2+jingdu_prest//2+1]
            print(data.shape)
            mode_output.append(data)
        output.append(mode_output)
    print(output)
    output = np.array(output)
    print()

    np.save(numpy_file, output)
    print('finish save!! next step gogogo')


if __name__ == '__main__':
    import numpy as np
    # 要处理的数据

    file_list = file_select('U://','txt')
    print(file_list)
    # ['U:///反对称1001_44个模式.txt',
    # 'U:///反对称y_对称z1001_42个模式.txt',
    # 'U:///对称y_反对称z1001_42个模式.txt',
    # 'U:///对称性1001_44个模式.txt']

    file_list=['data-01.txt']
    file_read = file_list[0]
    jingdu_preset = 51
    file_write = file_read.replace('.txt','delete_header.txt')
    numpy_saveplace = file_read.replace('.txt','.npy')
    mode_number= 3

    myfile = open(file_read, mode='r', encoding='utf-8')
    # print(myfile)
    data = myfile.readlines()
    start_line = find_start(data_list=data,sub_str='Ex')
    data = data[start_line:]

    with open(file_write, "w", ) as f:
        for i in range(len(data)):
            f.writelines(data[i])
    save_txt_to_numpy(txt_file=file_write,numpy_file=numpy_saveplace,mode_number=mode_number,
                      jingdu_prest=jingdu_preset)
    print('save_toNpy SUCCESS')

    # resize模式的数据
    file = numpy_saveplace
    a = np.load(file)
    file_resize_place = file.replace('.npy','_resize.npy')
    mode_number, fengliang, jingdu, jingdu01 = a.shape
    a_resize = np.reshape(a, (mode_number, fengliang, jingdu ** 2))
    np.save(file_resize_place, a_resize)


    # 保存到mat文件里面
    file_npy = numpy_saveplace
    matfile= numpy_saveplace.replace('.npy','_mat.mat')

    print('save to mat success! ')
    save_npy_to_mat(file_npy,matfile)

    import os
    os.remove(file_write)
