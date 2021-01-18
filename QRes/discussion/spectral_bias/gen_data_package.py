import numpy as np
import numpy.random as rn
from scipy.linalg import eigh
from scipy.linalg import norm
import time
#from scipy.linalg import qr

print('\n*** Importing gen_data.py')

def gen_xy(cfg, type):
    if type == 'train':
        n = cfg['n_train']
    elif type == 'val':
        n = cfg['n_val']
    else:
        print('bad type')
        return

    if cfg['gen_x_func'] == 'gen_x_circle_random':
        x, theta = gen_x_circle_random(n)
    elif cfg['gen_x_func'] == 'gen_x_circle_regular':
        x, theta = gen_x_circle_regular(n)
    elif cfg['gen_x_func'] == 'gen_x_uniform':
        x, theta = gen_x_uniform(n)
    elif cfg['gen_x_func'] == 'gen_x_circle_holes':
        x, theta = gen_x_circle_holes(n)

    W = rn.normal(0, 1, (cfg['dim'], cfg['n_units']))
    if cfg['gen_y_func'] == 'gen_y_fourier':
        y, vals = gen_y_fourier(theta, cfg['ks'], cfg['phases'])
    elif cfg['gen_y_func'] == 'gen_y_fourier_norm_1':
        y, vals = gen_y_fourier_norm_1(theta, cfg['ks'], cfg['phases'])
    elif cfg['gen_y_func'] == 'gen_y_H_inf':
        y, vals = gen_y_H_inf(x, cfg['ks'])
    elif cfg['gen_y_func'] == 'gen_y_H_inf_norm_1':
        y, vals = gen_y_H_inf_norm_1(x, cfg['ks'])
    elif cfg['gen_y_func'] == 'gen_y_H_0':
        y, vals, W = gen_y_H_0(x, cfg['ks'], W)

    return {'x' : x, 'y' : y, 'vals' : vals, 'W' : W, 'theta' : theta}

def gen_x_circle_random(n):
    theta = rn.uniform(-np.pi, np.pi, n)
    x = np.stack([np.cos(theta), np.sin(theta)]).T
    return x, theta

def gen_x_circle_regular(n):
    theta = np.linspace(-np.pi, np.pi, n+1)
    theta = theta[:-1]
    x = np.stack([np.cos(theta), np.sin(theta)]).T
    return x, theta

def gen_x_uniform(n):
    x = rn.uniform(-np.pi, np.pi, n)
    theta = x
    return x, theta

def gen_x_circle_holes(n):
    theta0 = rn.uniform(-np.pi, -np.pi/3, n // 2)
    theta1 = rn.uniform(np.pi/3, np.pi, n - n // 2)
    theta = np.concatenate((theta0, theta1), 0)
    x = np.stack([np.cos(theta), np.sin(theta)]).T
    return x, theta

def gen_y_fourier(theta, ks, phases = []):
    if phases == []:
        phases = np.zeros(len(ks))
    n = len(theta)
    y = np.zeros(n)
    for i_k, k in enumerate(ks):
        phi_k = phases[i_k]
        y = y + np.sin(k*(theta + phi_k))
    return y, ks

def gen_y_fourier_norm_1(theta, ks, phases = []):
    if phases == []:
        phases = np.zeros(len(ks))
    n = len(theta)
    y = np.zeros(n)
    for i_k, k in enumerate(ks):
        phi_k = phases[i_k]
        y = y + np.sin(k*(theta + phi_k))
    y = y / norm(y)
    return y, ks

def gen_y_H_inf(x, ks):
    if len(ks) > 1:
        print('`gen_y_H_inf` does not support yet more than one k')
        return
    k = ks[0]
    x = x.T
    n = x.shape[1]
    gram = np.matmul(x.T, x)
    gram = np.clip(gram, -1, 1)
    arcs = np.arccos(gram)
    H_inf = gram * (np.pi - arcs) / (2*np.pi)

    vals, vecs = eigh(H_inf, eigvals = (n-k-1, n-k-1))
    vec = vecs / norm(vecs) * np.sqrt(n)
    return vec.reshape(-1), vals

def gen_y_H_inf_norm_1(x, ks):
    if len(ks) > 1:
        print('`gen_y_H_inf` does not support yet more than one k')
        return
    k = ks[0]
    x = x.T
    n = x.shape[1]
    gram = np.matmul(x.T, x)
    gram = np.clip(gram, -1, 1)
    arcs = np.arccos(gram)
    H_inf = gram * (np.pi - arcs) / (2*np.pi)

    vals, vecs = eigh(H_inf, eigvals = (n-k-1, n-k-1))
    vec = vecs / norm(vecs)
    return vec.reshape(-1), vals

def gen_y_H_0(x, ks, W):
    if len(ks) > 1:
        print('`gen_y_H_0` does not support yet more than one k')
        return
    k = ks[0]
    x = x.T
    I = (np.sign(np.matmul(W.T, x)) + 1) / 2
    H_0 = np.matmul(x.T, x) * np.matmul(I.T, I) / W.shape[1]

    n = x.shape[1]
    vals, vecs = eigh(H_0, eigvals = (n-k-1, n-k-1))
    vec = vecs / norm(vecs) * np.sqrt(n)
    return vec.reshape(-1), vals, W
