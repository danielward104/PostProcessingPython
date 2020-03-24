import numpy as np
#import math
#from array import array
import sys
import matplotlib.pyplot as plt
# The following allows the script to work over SSH.  Check it still works when not SSH tunnelling.
plt.switch_backend('agg')
import os
#from tempfile import TemporaryFile
#from colorsys import hsv_to_rgb
#import pickle
#import copy
#import statistics as st
import scipy.interpolate as si

# Import user-defined modules (comp-pc6076 vs ARC vs VIPER).
#sys.path.insert(1,'/home/cserv1_a/soc_pg/scdrw/Documents/nbudocuments/PhD/SimNumerics/Python/postProcessingLib/')
sys.path.insert(1,'/home/home01/scdrw/Python')
#sys.path.insert(1,'/home/617122/Python/')
import readingNek as rn
import mappings as mp
import plottingTools as pt
import generalTools as tools

# Print whole arrays instead of truncating.
np.set_printoptions(threshold=sys.maxsize)


def pseudoColour( filename, order, dimension, start_file, jump, final_timestep, numelz ):

    fineness = 2

    order = order - 1

    final_file = int(final_timestep/jump)

    if (start_file == 1):
        range_vals = [x - (jump - 1) for x in np.array(range(1,final_file+1))*jump]

    else:
        range_vals = np.array(range(int(math.floor(float(start_file)/jump)),final_file+1))*jump
        print("Make sure calculation of file numbers is correct")
        print(range_vals)

    # Checking save-directory exists.
    foldername = "images_ps"
    folderexist = os.path.isdir(''.join(["./",foldername]))

    print(' ')
    if folderexist:
        print("The folder '",foldername,"' exists.")
    else:
        print("Creating the folder '",foldername,"'.")
        cwd = os.getcwd()
        os.mkdir(''.join([cwd,"/",foldername]))
    print(' ')

    # Reading in mesh data.
    data,time,istep,header,elmap,u_i,v_i,w_i,t_i,s_i = rn.readnek(''.join([filename,'0.f00001']))

    X = data[:,:,0]
    Y = data[:,:,1]
    Z = data[:,:,2]

    file_counter = 0

    # Looking only at specific vertical slice.
    y_slice = 0.0
    Y = np.array(Y)

    nel,gll = np.shape(Y)
    order = int(round(gll**(1/3),0))
    npoints = nel*gll

    coords1 = []
    coords2 = []
    test = 0.001

    ell = 0
    odd = 0
    for el in range(0,nel):
        ell_check = 1
        for od in range(0,gll):
            tester = abs(Y[el,od] - y_slice)
            if ( tester < test ):
                mapp = ell + odd
                coords1.append(el)
                coords2.append(od)
                if (ell_check > 0):
                    ell += 1
                    ell_check = 0
                else:
                    odd += 1

    minx, maxx = min(X[coords1,coords2]), max(X[coords1,coords2])
    minz, maxz = min(Z[coords1,coords2]), max(Z[coords1,coords2])

    print('Limits of domain (x,z):')
    print(minx,maxx,minz,maxz)
    print(' ')

    for k in range_vals:

        file_num = int((k-1)/jump + 1)

        # Outputs counter to terminal.
        if (start_file == 1):
            files_remaining = int(final_file - k/jump)
        else:
            files_remaining = int(final_file - (k-range_vals[0])/jump - start_file/jump)

        sys.stdout.write("\r")
        sys.stdout.write("Files remaining: {:3d}".format(files_remaining))
        sys.stdout.flush()

        file_counter += 1

        # Reads data files.
        data,time,istep,header,elmap,u_i,v_i,w_i,t_i,s_i = rn.readnek(''.join([filename, \
            '0.f',repr(k).zfill(5)]))

        S = data[:,:,s_i]

        Xplane = np.array(X[coords1,coords2])
        Zplane = np.array(Z[coords1,coords2])
        Splane = np.array(S[coords1,coords2])

        # Interpolate to uniform grid for plotting.
        zpoints = numelz*order
        npoints = len(Xplane)
        xpoints = npoints/zpoints
        xel = xpoints/order

        xi = np.linspace(minx,maxx,fineness*xpoints)
        zi = np.linspace(minz,maxz,fineness*zpoints)
        xi,zi = np.meshgrid(xi,zi)
        xzplane = np.zeros((npoints,2))
        xzplane[:,0] = Xplane
        xzplane[:,1] = Zplane

        Si = si.griddata(xzplane,Splane,(xi,zi),method='cubic')

        x_size = maxx - minx
        z_size = maxz - minz

        fig_x = 10
        fig_z = fig_x*(z_size/x_size)

        plt.figure(figsize=(fig_x,fig_z))
        plt.pcolor(xi,zi,Si,cmap='RdBu_r')
        plt.savefig(os.path.join(''.join(['./',foldername,'/passive_scalar',repr(file_counter).zfill(5),'.png'])),bbox_inches='tight')
        plt.close('all') 


    return
