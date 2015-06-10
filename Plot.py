import numpy as np
from numpy import linalg as la
from copy import deepcopy
import matplotlib.pyplot as plt


#plot first 5 things in each activity
#input raw data numpy file, labels numpy file, prefix of file to save to
#save pictures to files by prefix
def plot_raw(raw_file, y_file, prefix, num): 
    x = range(0,256,2)
    df = np.load(raw_file)
    y = np.load(y_file)
    for j in range(6):
        ts_list = df[y==j]
        if len(ts_list)==0:
            continue
        # acceleration
        plt.subplot(311)
        plt.ylabel('Acc x-axis $(m s^{-2})$')
        for i in range(num):
            plt.plot(x,ts_list[i][:, 0], ms=2)
        plt.subplot(312)
        plt.ylabel('Acc y-axis $(m s^{-2})$')
        for i in range(num):
            plt.plot(x,ts_list[i][:, 1], ms=2)
        plt.subplot(313)
        plt.ylabel('Acc z-axis $(m s^{-2})$')
        plt.xlabel('Time (ms)')
        for i in range(num):
            plt.plot(x,ts_list[i][:, 2], ms=2)
        plt.savefig(prefix+'_acc'+str(j))
        plt.close()
        # gyroscope
        plt.subplot(311)
        plt.ylabel('$\omega$ x-axis $(rad s^{-1})$')
        for i in range(num):
            plt.plot(x,ts_list[i][:, 3], ms=2)
        plt.subplot(312)
        plt.ylabel('$\omega$ y-axis $(rad s^{-1})$')
        for i in range(num):
            plt.plot(x,ts_list[i][:, 4], ms=2)
        plt.subplot(313)
        plt.ylabel('$\omega$ z-axis $(rad s^{-1})$')
        plt.xlabel('Time (ms)')
        for i in range(num):
            plt.plot(x,ts_list[i][:, 5], ms=2)
        plt.savefig(prefix+'_gyr'+str(j))
        plt.close()


#plot with multiple templates
#input numpy file containing templates, aligned numpy raw data, labels numpy file, prefix of file to save to
#save pictures to files by prefix
def plot_template_many(temp_file, align_file, y_file, prefix): # plot with multiple templates
    x = range(0,256,2)
    templates = np.load(temp_file+'.npy')
    df = np.load(align_file+'.npy')
    y = np.load(y_file+'.npy')
    for j in range(len(y)):
        label = y[j]
        first_index = sum(y<label)
        ts_list = df[j]
        nseries = len(ts_list) # number of samples
        dba_avg = templates[j]
        # acceleration
        plt.subplot(311)
        plt.ylabel('Acc x-axis $(m s^{-2})$')
        for i in range(nseries):
            ts_list[i] = np.array(ts_list[i])
            print ts_list[i][:,0]
            plt.plot(x,ts_list[i][:, 0], '.', ms=2)
        plt.plot(x,dba_avg[:, 0], 'k', linewidth=2)
        plt.subplot(312)
        plt.ylabel('Acc y-axis $(m s^{-2})$')
        for i in range(nseries):
            plt.plot(x,ts_list[i][:, 1], '.', ms=2)
        plt.plot(x,dba_avg[:, 1], 'k', linewidth=2)
        plt.subplot(313)
        plt.ylabel('Acc z-axis $(m s^{-2})$')
        plt.xlabel('Time (ms)')
        for i in range(nseries):
            plt.plot(x,ts_list[i][:, 2], '.', ms=2)
        plt.plot(x,dba_avg[:, 2], 'k', linewidth=2)
        plt.savefig(prefix+'_acc'+str(y[j])+' cluster'+str(j-first_index))
        plt.close()
        # gyroscope
        plt.subplot(311)
        plt.ylabel('$\omega$ x-axis $(rad s^{-1})$')
        for i in range(nseries):
            plt.plot(x,ts_list[i][:, 3], '.', ms=2)
        plt.plot(x,dba_avg[:, 3], 'k', linewidth=2)
        plt.subplot(312)
        plt.ylabel('$\omega$ y-axis $(rad s^{-1})$')
        for i in range(nseries):
            plt.plot(x,ts_list[i][:, 4], '.', ms=2)
        plt.plot(x,dba_avg[:, 4], 'k', linewidth=2)
        plt.subplot(313)
        plt.ylabel('$\omega$ z-axis $(rad s^{-1})$')
        plt.xlabel('Time (ms)')
        for i in range(nseries):
            plt.plot(x,ts_list[i][:, 5], '.', ms=2)
        plt.plot(x,dba_avg[:, 5], 'k', linewidth=2)
        plt.savefig(prefix+'_gyr'+str(y[j])+' cluster'+str(j-first_index))
        plt.close()