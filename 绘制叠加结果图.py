# 包括分量的强度图什么的
# -- coding: UTF-8 --
# 查看系数所得的结果图

# 新建文件保存图片
def makedir(dir):
    import os
    dir = dir.strip()
    isExist = os.path.exists(dir)
    if not isExist:
        os.makedirs(dir)
        return True
    else:
        return False


# 归一化到0-1 范围
def normalizationTobetween0and1(array):
    max_array = np.max(array)
    min_array = np.min(array)
    array_nor = (array - min_array) / (max_array - min_array) * 1.0
    return array_nor


# 特定的文件后缀保存，筛选后的特定后缀
def file_select(data_dir, *args):  #
    # data_dir is file_dir//
    # *args is 'png','jpeg','bmp' so on
    import glob
    # *args是要筛选的的后缀名称可以选择多个
    length = len(args)
    file_list = []
    for i in range(length):
        file_list = file_list + list(glob.glob(data_dir + '/*.' + str(args[i])))
    # file_list = list(glob.glob(data_dir + '/*.png')) + list(glob.glob(data_dir + '/*.jpg'))   # get name list of all .png files
    # data = []
    # print(file_list) # 得到文件的路径列表
    return file_list

def draw_single_fenglaing(array,jingdu=1001,ex_ey_ez='ex',saveplace='', cmap_style=''):

    plt.ion()
    e_middle=np.reshape(np.abs(array),(jingdu,jingdu) )
    plt.imshow(e_middle, cmap=cmap_style)
    plt.title(ex_ey_ez+'_amplitude')
    cbar = plt.colorbar()
    cbar.set_label('amplitude', rotation=-90, va='bottom')

    max_index = np.max(e_middle)
    min_index = np.min(e_middle)
    interval_temp = (max_index - min_index) / 5
    cbar.set_ticks([min_index, min_index + 1 * interval_temp,
                    min_index + 2 * interval_temp, min_index + 3 * interval_temp,
                    min_index + 4 * interval_temp, min_index + 5 * interval_temp])
    # set the font size of colorbar
    cbar.ax.tick_params(labelsize=8)

    plt.savefig(saveplace+ex_ey_ez+'_amplitude.png')
    plt.pause(0.005)
    plt.close('all')

    e_middle_x_length, e_middle_y_length = e_middle.shape

    esum_temp = np.resize(e_middle, (jingdu, jingdu))
    # 画出red 和black的图像
    for bilibli in range(1):
        for red in range(1):
            plt.ion()
            # red 保存
            # 寻找沿着y轴中心的画图方式
            x_label = np.arange(-1 * int(e_middle_y_length // 2), 1 * int(e_middle_y_length // 2) + 1)
            # 归一化吗?no

            nouse_red = np.array(
                esum_temp[int(e_middle_y_length // 2), :]
            )
            y_label = (
                np.abs(np.resize(nouse_red, (e_middle_y_length, 1)))
            )
            # pic_save_dir_red = pic_save_dir + '//red'

            save_place =saveplace+ex_ey_ez + '_mode_red' + '.png'

            plt.title('red-mode序号是：')
            plt.plot(x_label, y_label)
            plt.savefig(save_place)
            plt.show()
            plt.pause(0.005)
            plt.close('all')

        # 画出black
        for black in range(1):
            plt.ion()
            # 对黑线进行画图
            x_label = np.arange(-1 * int(e_middle_y_length // 2), 1 * int(e_middle_y_length // 2) + 1)
            print('xlabel is ', x_label, x_label.shape)
            # 归一化吗? no!
            nouse_black = np.array(
                esum_temp[:, e_middle_y_length // 2]
            )
            y_label = (
                np.abs(np.resize(nouse_black, (e_middle_x_length, 1)))
            )
            print('ylabel is ', y_label, y_label.shape)
            # pic_save_dir_black = pic_save_dir + '//black'

            save_place = saveplace+ex_ey_ez + '_mode_black' + '.png'

            plt.title('black-mode的序号是：')
            plt.plot(x_label, y_label)
            plt.savefig(save_place)
            plt.show()
            plt.pause(0.005)
            plt.close('all')

    # 沿着对角线01画出图像：
    for bibi in range(1):
        # 对角01
        plt.ion()
        # red 保存
        # 寻找沿着对角线轴中心的画图方式，点的个数都是相同的
        x_label = np.arange(-1 * int(e_middle_y_length // 2), 1 * int(e_middle_y_length // 2) + 1)
        # 归一化吗?no

        duijiaoDict = []
        for temp_number in range(e_middle_x_length):
            duijiaoDict.append(e_middle[temp_number, temp_number])
        nouse_duijiao = np.resize(duijiaoDict, (e_middle_x_length, 1))

        y_label = (
            nouse_duijiao
        )
        # pic_save_dir_duijiao = pic_save_dir + '//duijiao01'

        save_place = saveplace + ex_ey_ez + ''  + '_mode_duijiao01' + '.png'

        plt.title('duijiao01-mode序号是：')
        plt.plot(x_label, y_label)
        plt.savefig(save_place)
        plt.show()
        plt.pause(0.005)
        plt.close('all')

    # 沿着对角线02画出图像：
    for bibi in range(1):
        # 对角01
        plt.ion()
        # red 保存
        # 寻找沿着对角线轴中心的画图方式，点的个数都是相同的
        x_label = np.arange(-1 * int(e_middle_y_length // 2), 1 * int(e_middle_y_length // 2) + 1)

        # 归一化吗?no

        duijiaoDict = []
        for temp_number in range(jingdu):
            duijiaoDict.append(esum_temp[temp_number, e_middle_y_length - temp_number - 1])
        nouse_duijiao = np.resize(duijiaoDict, (e_middle_x_length, 1))

        y_label = (
            np.abs(nouse_duijiao)
        )

        # pic_save_dir_duijiao = pic_save_dir + '//duijiao02'

        save_place = saveplace+ex_ey_ez + '_duijiao02'+ '.png'

        plt.title('duijiao02-mode序号是：')
        plt.plot(x_label, y_label)
        plt.savefig(save_place)
        plt.show()
        plt.pause(0.005)
        plt.close('all')



if __name__ == '__main__':
    now_style = ['AA', 'AS', 'SA', 'SS']


    import matplotlib.pyplot as plt

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    import numpy as np

    now_style = '8x8fangzhenquyu_1001jingdu_resize'
    mode_save = '../../'+ now_style + '.npy'
    # 选择哪个系数文档 ---------------要修改-------------------------------------
    mode_sum = np.load(mode_save)

    k_save_dir = '..//k_to_save//' + now_style + '//'
    k_list = file_select(k_save_dir, 'npy')
    k_list = sorted(k_list, key=lambda i: len(i), reverse=False)
    print('k list is ', k_list)
    for k_one in k_list:
        k_to_load = k_one
        k_to_load = np.load(k_to_load)
        # k_save =k_one
        pic_save_dir = 'pic_to_save//1d_bigger//' + now_style + '//'
        print('save dir is ', pic_save_dir)
        makedir(pic_save_dir)
        k = k_to_load  # k为43，1的复数矩阵
        mode_number = k_to_load.shape[0]
        mode_fenliang = 3



        rows ,cols ,jingdu22= mode_sum.shape
        jingdu = int((max(cols ,rows,jingdu22))**0.5)
        middle_zuobiao = jingdu // 2
        count = 21

        banfengkuan_int =int(count*1.2)


        print(rows,cols,jingdu22)
        # 得到复数矩阵
        mode_fushu = np.zeros((mode_number, mode_fenliang, jingdu * jingdu), dtype=complex)  # 40 , 3 ,801**2
        # 得到复数矩阵 ， 【40，3，801*801】维度 , 0维度是ex ，1维度是ey ，2 维度是ez

        mode_fushu = mode_sum # 【40，3，1001*1001】维度


        # k shape  40 , 1 ()
        k_a,k_b = k.shape
        #v 结果为3 ， 801**2  维度
        mode_result = np.zeros((mode_fenliang, jingdu**2 ) , dtype=complex)
        for everypoint in range(jingdu**2):
            for fenliang in range(mode_fenliang):
                for i in range(mode_number):
                    mode_result[fenliang,everypoint] += k[i,0] * mode_fushu[i,fenliang,everypoint]

        # 3 , 801**2
        mode_result =mode_result
        fengliang_saveplace=pic_save_dir+'分量强度保存地址//'
        makedir(fengliang_saveplace)
        for _ in range(1):
            # draw single fenliang intensity

            draw_single_fenglaing(array=mode_result[0, :],
                                  ex_ey_ez='ex',
                                  saveplace=fengliang_saveplace , cmap_style='jet')
            draw_single_fenglaing(array=mode_result[1, :],
                                  ex_ey_ez='ey',
                                  saveplace=fengliang_saveplace , cmap_style='jet')
            draw_single_fenglaing(array=mode_result[0+1+1, :],
                                  ex_ey_ez='ez',
                                  saveplace=fengliang_saveplace , cmap_style='jet')

        mode_pic = np.zeros((1,jingdu**2),dtype=float)
        # 将e变为1 ，801**2
        for point in range(jingdu**2):
            for i in range(mode_fenliang):
                mode_pic[0, point] = mode_pic[0, point] + (np.abs(mode_result[i,point]))**2
        mode_pic =mode_pic

        # 叠加的结果 去掉绝对值
        esum_temp = np.resize(np.abs(mode_pic), (jingdu, jingdu))
        esum_temp = normalizationTobetween0and1(esum_temp)
        e_middle = esum_temp[middle_zuobiao-banfengkuan_int*2:middle_zuobiao+banfengkuan_int*2+1,middle_zuobiao-banfengkuan_int*2:middle_zuobiao+banfengkuan_int*2+1]
        e_middle_x_length,e_middle_y_length   =e_middle.shape
        plt.ion()
        plt.imshow(e_middle, cmap='plasma')
        plt.title('ex的第' + '个解，所采用的系数对所有模式叠加得到的强度结果图')
        cbar = plt.colorbar()
        cbar.set_label('intensity', rotation=-90, va='bottom')

        max_index = np.max(e_middle)
        min_index = np.min(e_middle)
        interval_temp = ( max_index - min_index ) / 5
        cbar.set_ticks([min_index, min_index + 1 * interval_temp,
                        min_index + 2 * interval_temp, min_index + 3 * interval_temp,
                        min_index + 4 * interval_temp, min_index + 5 * interval_temp])
        # set the font size of colorbar
        cbar.ax.tick_params(labelsize=8)
        save_place = pic_save_dir + '//' + 'ex' + '.png'
        plt.savefig(save_place)
        plt.pause(0.005)
        plt.close('all')

        # 画出red 和black的图像
        for bilibli in range(1):
            for red in range(1):
                plt.ion()
                # red 保存
                # 寻找沿着y轴中心的画图方式
                x_label = np.arange(-1 * int(e_middle_y_length//2),1 * int(e_middle_y_length//2) +1)
                # 归一化吗?no

                nouse_red = np.array(
                    e_middle[int(e_middle_y_length//2),:]
                )
                y_label = (
                    np.abs(np.resize(nouse_red, (e_middle_y_length, 1)))
                )
                # pic_save_dir_red = pic_save_dir + '//red'
                pic_save_dir_red = pic_save_dir
                makedir(pic_save_dir_red)
                save_place = pic_save_dir_red + '//' + 'mode_red' + '.png'

                plt.title('red-mode序号是：')
                plt.plot(x_label, y_label)
                plt.savefig(save_place)
                plt.show()
                plt.pause(0.005)
                plt.close('all')

            # 画出black
            for black in range(1):
                plt.ion()
                # 对黑线进行画图
                x_label =  np.arange(-1 * int(e_middle_y_length//2),1 * int(e_middle_y_length//2) +1)
                print('xlabel is ', x_label, x_label.shape)
                # 归一化吗? no!
                nouse_black = np.array(
                    e_middle[  : ,e_middle_y_length//2 ]
                )
                y_label = (
                    np.abs(np.resize(nouse_black , (e_middle_x_length, 1)))
                )
                print('ylabel is ', y_label, y_label.shape)
                # pic_save_dir_black = pic_save_dir + '//black'
                pic_save_dir_black = pic_save_dir
                makedir(pic_save_dir_black)

                save_place = pic_save_dir_black + '//' + 'mode_black' + '.png'

                plt.title('black-mode的序号是：')
                plt.plot(x_label, y_label)
                plt.savefig(save_place)
                plt.show()
                plt.pause(0.005)
                plt.close('all')

        # 沿着对角线01画出图像：
        for bibi in range(1):
            # 对角01
            plt.ion()
            # red 保存
            # 寻找沿着对角线轴中心的画图方式，点的个数都是相同的
            x_label = np.arange(-1 * int(e_middle_y_length//2),1 * int(e_middle_y_length//2) +1)
            # 归一化吗?no

            duijiaoDict = []
            for temp_number in range(e_middle_x_length):
                duijiaoDict.append(e_middle[temp_number, temp_number])
            nouse_duijiao = np.resize(duijiaoDict, (e_middle_x_length, 1))

            y_label = (
                nouse_duijiao
            )
            # pic_save_dir_duijiao = pic_save_dir + '//duijiao01'
            pic_save_dir_duijiao = pic_save_dir
            makedir(pic_save_dir_duijiao)
            save_place = pic_save_dir_duijiao + '//' + 'mode_duijiao01' + '.png'

            plt.title('duijiao01-mode序号是：')
            plt.plot(x_label, y_label)
            plt.savefig(save_place)
            plt.show()
            plt.pause(0.005)
            plt.close('all')

        # 沿着对角线02画出图像：
        for bibi in range(1):
            # 对角01
            plt.ion()
            # red 保存
            # 寻找沿着对角线轴中心的画图方式，点的个数都是相同的
            x_label = np.arange(-1 * int(e_middle_y_length//2),1 * int(e_middle_y_length//2) +1)

            # 归一化吗?no

            duijiaoDict = []
            for temp_number in range(jingdu):
                duijiaoDict.append(esum_temp[temp_number, e_middle_y_length - temp_number - 1])
            nouse_duijiao = np.resize(duijiaoDict, (e_middle_x_length, 1))

            y_label = (
                np.abs(nouse_duijiao)
                       )


            # pic_save_dir_duijiao = pic_save_dir + '//duijiao02'
            pic_save_dir_duijiao = pic_save_dir
            makedir(pic_save_dir_duijiao)
            save_place = pic_save_dir_duijiao + '//' + 'mode_duijiao02' + '.png'

            plt.title('duijiao02-mode序号是：')
            plt.plot(x_label, y_label)
            plt.savefig(save_place)
            plt.show()
            plt.pause(0.005)
            plt.close('all')



