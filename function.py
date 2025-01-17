import math
import pandas as pd
from sklearn.preprocessing import scale
from scipy.interpolate import griddata
from importlib import reload
from scipy.io import loadmat
from utils_TFF import makePath, cart2sph, pol2cart
import numpy as np

def get_data_from_mat(mat_path):
    '''
    discription:load data from mat path and reshape
    param{type}:mat_path: Str
    return{type}: onesub_data
    '''
    mat_eeg_data = []
    # mat_wavA_data = []
    # mat_wavB_data =[]
    mat_event_data=[]
    matstruct_contents = loadmat(mat_path)
    matstruct_contents = matstruct_contents['data']
    mat_event = matstruct_contents[0, 0]['event']['eeg'].item()
    mat_event_value = mat_event[0]['value']     # 1*60 1=male, 2=female
    mat_eeg = matstruct_contents[0, 0]['eeg'] # 60 trials 3200*66
    # mat_wavA = matstruct_contents[0, 0]['wavA']
    # mat_wavB = matstruct_contents[0, 0]['wavB']
    for i in range(mat_eeg.shape[1]):   #test.shape[1]的值是60
        mat_eeg_data.append(mat_eeg[0,i])
        # mat_wavA_data.append(mat_wavA[0,i])
        # mat_wavB_data.append(mat_wavB[0,i])
        mat_event_data.append(mat_event_value[i][0][0])

    return mat_eeg_data, mat_event_data

def azim_proj(pos):
    """
    Computes the Azimuthal Equidistant Projection of input point in 3D Cartesian Coordinates.
    Imagine a plane being placed against (tangent to) a globe. If
    a light source inside the globe projects the graticule onto
    the plane the result would be a planar, or azimuthal, map
    projection.

    :param pos: position in 3D Cartesian coordinates
    :return: projected coordinates using Azimuthal Equidistant Projection
    """
    [r, elev, az] = cart2sph(pos[0], pos[1], pos[2])
    return pol2cart(az, math.pi / 2 - elev)


def gen_images(data, args):
    locs = loadmat('locs_aved_orig.mat')
    locs_3d = locs['data']
    locs_2d = []
    for e in locs_3d:
        locs_2d.append(azim_proj(e))

    locs_2d_final = np.array(locs_2d)
    grid_x, grid_y = np.mgrid[
                     min(np.array(locs_2d)[:, 0]):max(np.array(locs_2d)[:, 0]):args.image_size * 1j,
                     min(np.array(locs_2d)[:, 1]):max(np.array(locs_2d)[:, 1]):args.image_size * 1j]

    images = []
    for i in range(data.shape[0]):
        images.append(griddata(locs_2d_final, data[i, :], (grid_x, grid_y), method='cubic', fill_value=np.nan))
    images = np.stack(images, axis=0)

    images[~np.isnan(images)] = scale(images[~np.isnan(images)])
    images = np.nan_to_num(images)
    return images


def read_prepared_data(args):
    data = []

    for l in range(len(args.ConType)):
        label = pd.read_csv(args.data_document_path + "/csv/" + args.name + args.ConType[l] + ".csv")
        target = []
        for k in range(args.trail_number):
            filename = args.data_document_path + "/" + args.ConType[l] + "/" + args.name + "Tra" + str(k + 1) + ".csv"
            #KUL_single_single3,contype=no,name=s1,len(arg.ConType)=1
            data_pf = pd.read_csv(filename, header=None)
            eeg_data = data_pf.iloc[:, 2:] #KUL,DTU
            # eeg_data = data_pf.iloc[64:, :] #PKU

            data.append(eeg_data)
            target.append(label.iloc[k, args.label_col])


    return data, target

def sliding_window(eeg_datas, labels, args, out_channels):
    window_size = args.window_length
    stride = int(128 * (1 - args.overlap))

    train_eeg = []
    test_eeg = []
    train_label = []
    test_label = []

    for m in range(len(labels)): #labels 0-19
        eeg = eeg_datas[m]
        label = labels[m]
        windows = []
        new_label = []
        for i in range(0, eeg.shape[0] - window_size + 1, stride):
            window = eeg[i:i+window_size, :]
            windows.append(window)
            new_label.append(label)

        if m in [14,15]:
            test_eeg.append(np.array(windows))
            test_label.append(np.array(new_label))
        else:
            train_eeg.append(np.array(windows))
            train_label.append(np.array(new_label))

    train_eeg = np.stack(train_eeg, axis=0).reshape(-1, window_size, out_channels)
    test_eeg = np.stack(test_eeg, axis=0).reshape(-1, window_size, out_channels)
    train_label = np.stack(train_label, axis=0).reshape(-1, 1)
    test_label = np.stack(test_label, axis=0).reshape(-1, 1)

    return train_eeg, test_eeg, train_label, test_label

def to_alpha0(data, args):
    alpha_data = []
    for window in data:
        window_data0 = np.fft.fft(window, n=args.window_length, axis=0)
        window_data0 = np.abs(window_data0)
        window_data0 = np.sum(np.power(window_data0[args.point0_low:args.point0_high, :], 2), axis=0)
        window_data0 = np.log2(window_data0 / args.window_length)
        alpha_data.append(window_data0)
    alpha_data = np.stack(alpha_data, axis=0)
    return alpha_data

def to_alpha1(data, args):
    alpha_data = []
    for window in data:
        window_data1 = np.fft.fft(window, n=args.window_length, axis=0)
        window_data1 = np.abs(window_data1)
        window_data1 = np.sum(np.power(window_data1[args.point1_low:args.point1_high, :], 2), axis=0)
        window_data1 = np.log2(window_data1 / args.window_length)
        alpha_data.append(window_data1)
    alpha_data = np.stack(alpha_data, axis=0)
    return alpha_data

def to_alpha2(data, args):
    alpha_data = []
    for window in data:
        window_data2 = np.fft.fft(window, n=args.window_length, axis=0)
        window_data2 = np.abs(window_data2)
        window_data2 = np.sum(np.power(window_data2[args.point2_low:args.point2_high, :], 2), axis=0)
        window_data2 = np.log2(window_data2 / args.window_length)
        alpha_data.append(window_data2)
    alpha_data = np.stack(alpha_data, axis=0)
    return alpha_data

def to_alpha3(data, args):
    alpha_data = []
    for window in data:
        window_data3 = np.fft.fft(window, n=args.window_length, axis=0)
        window_data3 = np.abs(window_data3)
        window_data3 = np.sum(np.power(window_data3[args.point3_low:args.point3_high, :], 2), axis=0)
        window_data3 = np.log2(window_data3 / args.window_length)
        alpha_data.append(window_data3)
    alpha_data = np.stack(alpha_data, axis=0)
    return alpha_data

def to_alpha4(data, args):
    alpha_data = []
    for window in data:
        window_data4= np.fft.fft(window, n=args.window_length, axis=0)
        window_data4 = np.abs(window_data4)
        window_data4 = np.sum(np.power(window_data4[args.point4_low:args.point4_high, :], 2), axis=0)
        window_data4 = np.log2(window_data4 / args.window_length)
        alpha_data.append(window_data4)
    alpha_data = np.stack(alpha_data, axis=0)
    return alpha_data