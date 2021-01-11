import pandas as pd
import numpy as np
import time


# 只筛选了ey+ ez的数据
# 模型信息，几个mode ，精度多少


def nor0TO1(array):
    maxis = np.max(array)
    minis = np.min(array)
    rows, cols = array.shape
    array_new = np.zeros((rows, cols))
    for row in range(rows):
        for col in range(cols):
            if array[row, col] >= 0:
                array_new[row, col] = (array[row, col] - 0.0) / (maxis - 0.0) * 1.0
            if array[row, col] < 0:
                array_new[row, col] = (0.0 - array[row, col]) / (0.0 - minis) * (-1.0)
    return array_new

def makedir(dir):
    import os
    dir = dir.strip()
    isExist = os.path.exists(dir)
    if not isExist:
        os.makedirs(dir)
        return True
    else:
        return False

def savename(temp, number):
    result = ''
    for i in range(number):
        j = 1 + i
        temp01 = temp.replace('1', str(j))
        result = result + (temp01) + ','
    return result

# 得到所有的columns name in pandas ,取前number个模式
def getColumns(number):
    temp = r"'mode1_Ex(real)','mode1_Ex(imag)','mode1_Ey(real)','mode1_Ey(imag)','mode1_Ez(real)','mode1_Ez(imag)'"
    result = savename(temp=temp, number=number)
    print(result)
    return result
# 归一化到0-1 范围
def normalizationTobetween0and1(array):
    max_array = np.max(array)
    min_array = np.min(array)
    array_nor = (array - min_array) / (max_array - min_array) * 1.0
    return array_nor



def saveArrayToPicture(array,jingdu=201, mode_number=80, e_pic='epic//',pic_title='mode_sum'):
    import matplotlib.pyplot as plt
    makedir(e_pic)
    e_sum = (array)

    max_index = np.max(e_sum)
    min_index = np.min(e_sum)
    interval = (max_index - min_index) / 5
    e_sum = np.resize(e_sum, (jingdu, jingdu))

    a,b = e_sum.shape

    # aa= e_sum[:a//2,:]
    # bb= e_sum[a//2+1:,:]
    # print(aa.shape)
    # print(bb.shape)
    # print(aa-bb)
    cou=0

    for i in range(a//2):
        for j in range(b):
            t1 = e_sum[j,a//2-i]
            t2 = e_sum[j,a//2+i]
            cou=cou+1
            print(cou,'  ',np.abs(t1-t2))

    print(e_sum.shape)
    plt.ion()
    plt.title(pic_title + str(mode_number+ 1))
    plt.imshow(((e_sum)), cmap='plasma')

    cbar = plt.colorbar()
    cbar.set_label('intensity', rotation=-90, va='bottom')
    # cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    # set the font size of colorbar
    cbar.set_ticks([min_index, min_index + 1 * interval,
                    min_index + 2 * interval, min_index + 3 * interval,
                    min_index + 4 * interval, min_index + 5 * interval])
    cbar.ax.tick_params(labelsize=8)

    saveplace = e_pic + '//' + str(mode_number + 1) + 'mode_sum'
    plt.savefig(saveplace)

    plt.pause(.1)
    plt.close('all')


# def drawRedAndBlackAndDuijiao(array ,jingdu=801,pic_save_dir = 'save'):
#
#     esum_temp=np.resize(array,(jingdu , jingdu))
#     # 画出red 和black的图像
#     for bilibli in range(1):
#         plt.ion()
#         # red 保存
#         # 寻找沿着y轴中心的画图方式
#         x_label = np.arange(-1 * ((jingdu) // 4), (jingdu) // 4 + 1)
#         # 归一化吗?no
#
#         nouse_red = (
#             esum_temp[jingdu // 2, jingdu//2 - jingdu//4: jingdu//2 +jingdu//4+1]
#         )
#         y_label = (
#             np.abs(np.resize(nouse_red, (jingdu//2 +1 , 1)))
#         )
#         pic_save_dir_red = pic_save_dir + '//red'
#         makedir(pic_save_dir_red)
#         save_place = pic_save_dir_red + '//' + str(i) + 'mode_red' + '.png'
#
#         plt.title('red-mode序号是：' + str(i))
#         plt.plot(x_label, y_label)
#         plt.savefig(save_place)
#         plt.show()
#         plt.pause(0.05)
#         plt.close('all')
#
#         plt.ion()
#         # 对黑线进行画图
#         x_label = np.arange(-1 * ((jingdu) // 4), (jingdu) // 4 + 1)
#         print('xlabel is ', x_label, x_label.shape)
#         # 归一化吗? no!
#         nouse_black = (
#             esum_temp[jingdu//2 - jingdu//4: jingdu//2 +jingdu//4+1, jingdu // 2]
#         )
#         y_label = (
#             np.abs(np.resize(nouse_black, (jingdu//2+1, 1)))
#         )
#         print('ylabel is ', y_label, y_label.shape)
#         pic_save_dir_black = pic_save_dir + '//black'
#         makedir(pic_save_dir_black)
#
#         makedir(pic_save_dir)
#         save_place = pic_save_dir_black + '//' + str(i) + 'mode_black' + '.png'
#
#         plt.title('black-mode的序号是：' + str(i))
#         plt.plot(x_label, y_label)
#         plt.savefig(save_place)
#         plt.show()
#         plt.pause(0.05)
#         plt.close('all')
#
#     # 沿着对角线01画出图像：
#     for bibi in range(1):
#         # 对角01
#         plt.ion()
#         # red 保存
#         # 寻找沿着对角线轴中心的画图方式，点的个数都是相同的
#         x_label = np.arange(-1 * ((jingdu) // 4), (jingdu) // 4 + 1)
#         # 归一化吗?no
#
#         duijiaoDict = []
#         for temp_number in range(jingdu):
#             duijiaoDict.append(esum_temp[temp_number, temp_number])
#         nouse_duijiao = np.resize(duijiaoDict, (jingdu, 1))
#         nouse_duijiao = nouse_duijiao[jingdu//2 - jingdu//4 : jingdu//2 +jingdu//4 +1]
#
#         y_label = (
#             np.abs(np.resize(nouse_duijiao, (jingdu//2+1, 1)))
#         )
#         pic_save_dir_duijiao = pic_save_dir + '//duijiao01'
#         makedir(pic_save_dir_duijiao)
#         save_place = pic_save_dir_duijiao + '//' + str(i) + 'mode_duijiao01' + '.png'
#
#         plt.title('duijiao01-mode序号是：' + str(i))
#         plt.plot(x_label, y_label)
#         plt.savefig(save_place)
#         plt.show()
#         plt.pause(0.05)
#         plt.close('all')
#
#     # 沿着对角线02画出图像：
#     for bibi in range(1):
#         # 对角01
#         plt.ion()
#         # red 保存
#         # 寻找沿着对角线轴中心的画图方式，点的个数都是相同的
#         x_label = np.arange(-1 * ((jingdu) // 4), (jingdu) // 4 + 1)
#         # 归一化吗?no
#
#         duijiaoDict = []
#         for temp_number in range(jingdu):
#             duijiaoDict.append(esum_temp[temp_number, jingdu - temp_number - 1])
#         nouse_duijiao = np.resize(duijiaoDict, (jingdu, 1))
#         nouse_duijiao = nouse_duijiao[jingdu // 2 - jingdu // 4  :  jingdu // 2 + jingdu // 4 + 1]
#
#         y_label = (
#             np.abs(np.resize(nouse_duijiao, (jingdu//2+1, 1)))
#         )
#         pic_save_dir_duijiao = pic_save_dir + '//duijiao02'
#         makedir(pic_save_dir_duijiao)
#         save_place = pic_save_dir_duijiao + '//' + str(i) + 'mode_duijiao02' + '.png'
#
#         plt.title('duijiao02-mode序号是：' + str(i))
#         plt.plot(x_label, y_label)
#         plt.savefig(save_place)
#         plt.show()
#         plt.pause(0.05)
#         plt.close('all')
#


if __name__ == '__main__':
    mode_number = 40
    # j
    pic_dir = 'pic_origin_1000_1000'
    e_pic = pic_dir+'//e_pic'
    mode_file =r'./40mode_新的数据.txt'
    makedir(pic_dir)

    import numpy as np
    import matplotlib.pyplot as plt
    mode_80 = np.load('data-01.npy')

    jingdu = int((mode_80.shape[3]))
    saveArrayToPicture(array=(np.abs(mode_80[0,0,:,:]))**2
                             +(np.abs(mode_80[0,1,:,:]))**2
                               +(np.abs(mode_80[0,2,:,:]))**2
                               ,jingdu=jingdu,
                               mode_number=0,e_pic=e_pic+'//esum',pic_title='mode_sum')
    saveArrayToPicture(array=(np.abs(mode_80[0, 0, :, :])) ** 2

                       , jingdu=jingdu,
                       mode_number=1, e_pic=e_pic + '//esum', pic_title='mode_sum')
