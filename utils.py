from __future__ import print_function
import numpy as np
import math
np.set_printoptions(suppress=True)
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

def normalize_A(A,lmax=2):
    A=F.relu(A)
    N=A.shape[0]
    A=A*(torch.ones(N,N).cuda()-torch.eye(N,N).cuda())
    A=A+A.T
    d = torch.sum(A, 1)
    d = 1 / torch.sqrt((d + 1e-10))
    D = torch.diag_embed(d)
    L = torch.eye(N,N).cuda()-torch.matmul(torch.matmul(D, A), D)
    Lnorm=(2*L/lmax)-torch.eye(N,N).cuda()
    return Lnorm


def generate_cheby_adj(L, K):
    support = []
    for i in range(K):
        if i == 0:
            support.append(torch.eye(L.shape[-1]).cuda())
        elif i == 1:
            support.append(L)
        else:
            temp = torch.matmul(2*L,support[-1],)-support[-2]
            support.append(temp)
    return support


def cart2sph(x, y, z):
    """
    Transform Cartesian coordinates to spherical
    :param x: X coordinate
    :param y: Y coordinate
    :param z: Z coordinate
    :return: radius, elevation, azimuth
    """
    x2_y2 = x**2 + y**2
    r = math.sqrt(x2_y2 + z**2)                    # r
    elev = math.atan2(z, math.sqrt(x2_y2))            # Elevation
    az = math.atan2(y, x)                          # Azimuth
    return r, elev, az


def pol2cart(theta, rho):
    """
    Transform polar coordinates to Cartesian
    :param theta: angle value
    :param rho: radius value
    :return: X, Y
    """
    return rho * math.cos(theta), rho * math.sin(theta)


def makePath(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return path


def monitor(process, multiple, second):
    while True:
        sum = 0
        for ps in process:
            if ps.is_alive():
                sum += 1
        if sum < multiple:
            break
        else:
            time.sleep(second)

def save_load_name(args, name=''):
    name = name if len(name) > 0 else 'default_model'
    return name

def save_model(args, model, name=''):
    name = save_load_name(args, name)
    torch.save(model, f'/M-DBPNet/pre_trained_models/{name}_{args.window_length}.pt')


def load_model(args, name=''):
    name = save_load_name(args, name)
    model = torch.load(f'/M-DBPNet/pre_trained_models/{name}_{args.window_length}.pt')
    return model