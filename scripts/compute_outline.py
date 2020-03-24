import numpy as np
import math
#from array import array
import sys
import matplotlib.pyplot as plt
# The following allows the script to work over SSH.  Check it still works when not SSH tunnelling.
plt.switch_backend('agg')
import os
#from tempfile import TemporaryFile
#from colorsys import hsv_to_rgb
#import pickle
import copy
import statistics as st
import scipy.interpolate as si

# Import user-defined modules (comp-pc6076 vs ARC vs VIPER).
#sys.path.insert(1,'/home/cserv1_a/soc_pg/scdrw/Documents/nbudocuments/PhD/SimNumerics/Python/postProcessingLib/')      # Comp-pc6076
sys.path.insert(1,'/home/home01/scdrw/Python')          # ARC
#sys.path.insert(1,'/home/617122/Python/')          # VIPER
import readingNek as rn
import mappings as mp
import plottingTools as pt
import generalTools as tools

# Print whole arrays instead of truncating.
np.set_printoptions(threshold=sys.maxsize)


def umbrellaOutline( filename, order, dimension, start_file, jump, final_timestep, image_OnOff ):

    order = order - 1

    final_file = int(final_timestep/jump)

    if (start_file == 1):
        range_vals = [x - (jump - 1) for x in np.array(range(1,final_file+1))*jump]

    else:
        range_vals = np.array(range(int(math.floor(float(start_file)/jump)),final_file+1))*jump
        print("Make sure calculation of file numbers is correct")
        print(range_vals)


    # Reading in mesh data.
    data,time,istep,header,elmap,u_i,v_i,w_i,t_i,s_i = rn.readnek(''.join([filename,'0.f00001']))

    X = data[:,:,0]
    Y = data[:,:,1]
    Z = data[:,:,2]

    front_data_mean = []
    front_data_time = []

    for k in range_vals:

        file_num = int((k-1)/jump + 1)

        # Outputs counter to terminal.
        if (start_file == 1):
            files_remaining = int(final_file - k/jump)
        else:
            files_remaining = int(final_file - (k-range_vals[0])/jump - start_file/jump)

        sys.stdout.write("\r")
        sys.stdout.write("Files remaining: {:2d}".format(files_remaining))
        sys.stdout.flush()

        # Reads data files.
        data,time,istep,header,elmap,u_i,v_i,w_i,t_i,s_i = rn.readnek(''.join([filename, \
            '0.f',repr(k).zfill(5)]))

        S = data[:,:,s_i]

        [S_x, S_y] = S.shape

#        newS = []
        newX = []
        newY = []
        newZ = []

        for x in range(S_x):
            for y in range(S_y):
                if (S[x,y] > 0.01):
#                    newS.append(S[x,y]) 
                    newX.append(X[x,y])
                    newY.append(Y[x,y])
                    newZ.append(Z[x,y])


#        sliceS = []
#        sliceX = []
#        sliceY = []
        sliceR = []
        sliceth = []

        z_slice = 3.2
        test = np.asarray(newZ)
        idx = (np.abs(test - z_slice)).argmin()
        z_slice = test[idx]

        for i in range(len(newX)):
            if (newZ[i] == z_slice):
#                sliceS.append(newS[i])
#                sliceX.append(newX[i])
#                sliceY.append(newY[i])

                r = (newX[i]**2 + newY[i]**2)**0.5

                if (newX[i] > 0):
                    theta = math.atan(newY[i]/newX[i])
                elif (newX[i] < 0 and newY[i] >= 0):
                    theta = math.atan(newY[i]/newX[i]) + math.pi
                elif (newX[i] < 0 and newY[i] < 0):
                    theta = math.atan(newY[i]/newX[i]) - math.pi
                elif (newX[i] == 0 and newY[i] > 0):
                    theta = math.pi/2
                elif (newX[i] == 0 and newY[i] < 0):
                    theta = math.pi/2
                elif (newX[i] == 0 and newY[i] == 0):
                    theta = 0.0

                if (r > 0.15):
                    sliceR.append(round(r,3))
                    sliceth.append(round(theta,3))

        print(theta)

        # Sort both sliceth and sliceR at the same time.
        zipped = zip(sliceth,sliceR)
        zipped = sorted(zipped, key=lambda x: x[0])
        sliceth,sliceR = zip(*zipped)

        bigR = []
        bigth = []
        savepi = []

        # Group sets of theta, and find largest radius in that set for all sets.
        counter = 1
        piFlag = 0      # piFlag corrects the overlap between pi and -pi
        for i in range(len(sliceth)-1):
            if (sliceth[i] == sliceth[i+1]):
                counter = counter + 1
            else:
                if (piFlag == 0):
                    savepi.extend(sliceR[i-counter+1:i+1])
                else:
                    bigR.append(max(sliceR[i-counter+1:i+1]))
                    bigth.append(sliceth[i])
                    counter = 1
                piFlag = 1
            if (i == len(sliceth)-2):
                savepi.extend(sliceR[i-counter+1:i+1])
                bigR.append(max(savepi))
                bigth.append(sliceth[i])

        front_data_mean.append(st.mean(bigR))
        front_data_time.append(time)

    ##############################################################
    ######################### Plotting ###########################
    ##############################################################


        if (image_OnOff == 1):
#            fig = plt.figure(figsize=(6,5.8))
            fig = plt.figure(figsize=(12,11.6))   
 
            ax = fig.add_subplot(111, projection='polar')
            ax.scatter(sliceth,sliceR)
    
            ax.scatter(bigth,bigR,c='red')

            ax.set_ylim(0,6)

            plt.savefig(os.path.join(''.join(['./Images/scatter2D_',repr(file_num).zfill(5),'.png'])),\
                bbox_inches='tight')

            plt.close('all')

#    from mpl_toolkits.mplot3d import Axes3D
# 
#    fig = plt.figure(figsize=(20,20))
#    ax = fig.add_subplot(111,projection='3d')
#
#    ax.scatter(newX,newY,newZ)
#
#    plt.savefig(os.path.join(''.join(['scatter3D.png'])),bbox_inches='tight')
#
#    plt.close('all')
 
    print(front_data_mean,front_data_time)

    f = open('front_data_mean.file','w')
    for x in front_data_mean:
        f.write(str(x))
        f.write("\n")
    f.close()

    f = open('front_data_time.file','w')
    for x in front_data_time:
        f.write(str(x))
        f.write("\n")
    f.close()

    fig = plt.figure(figsize=(12,6))
    plt.plot(front_data_time,front_data_mean)

    plt.savefig(os.path.join(''.join(['front_data.png'])),bbox_inches='tight')

    plt.close('all')

    return




################################################################################
################################################################################
############################# Plume Outline ####################################
################################################################################
################################################################################




def plumeOutline( filename, order, dimension, start_file, jump, final_timestep, numelz, s_val, start_avg_at_time, image_OnOff ):

    plot_onImages = 1
    fineness = 1
    plot_frequency = 10

    if (plot_onImages == 1):
        print('Producing images for verification.  Will increase simulation time.')
        print(' ')

    top_data = []
    top_time = []

    order = order - 1

    final_file = int(final_timestep/jump)

    if (start_file == 1):
        range_vals = [x - (jump - 1) for x in np.array(range(1,final_file+1))*jump]

    else:
        range_vals = np.array(range(int(math.floor(float(start_file)/jump)),final_file+1))*jump
        print("Make sure calculation of file numbers is correct")
        print(range_vals)

    # Checking save-directory exists.
    foldername = "images_outline"
    folderexist = os.path.isdir(''.join(["./",foldername]))

    if folderexist:
        print("The folder '",foldername,"' exists.")
    else:
        print("Creating the folder '",foldername,"'.")
        cwd = os.getcwd()
        os.mkdir(''.join([cwd,"/",foldername]))
    print(' ')

    # Reading in mesh data.
    data,time,istep,header,elmap,u_i,v_i,w_i,t_i,s_i = rn.readnek(''.join([filename,'0.f00001']))

    Z = data[:,:,2]

    if (plot_onImages == 1):

        X = data[:,:,0]
        Y = data[:,:,1]
   
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

        minx, maxx = round(min(X[coords1,coords2]),0), round(max(X[coords1,coords2]),0)
        minz, maxz = min(Z[coords1,coords2]), max(Z[coords1,coords2])


    file_counter = 0
    print_counter = 0

    # Opening files for writing.
    f = open('top_data.file','w')
    g = open('top_time.file','w')

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

        [S_x, S_y] = S.shape

        newZ = []

        for x in range(S_x):
            for y in range(S_y):
                if (S[x,y] > s_val):
                    newZ.append(round(Z[x,y],2))

        maxPlumeHeight = max(newZ)
        top_data.append(maxPlumeHeight)
        top_time.append(time)

        # Writing to file.
        f.write(str(maxPlumeHeight))
        f.write("\n")
        g.write(str(time))
        g.write("\n")

        if ( file_counter % plot_frequency == 0 ):

            if (plot_onImages == 1):
                print_counter += 1

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

                plt.figure(figsize=(10,10))
                plt.pcolor(xi,zi,Si,cmap='RdBu_r')
                
                plt.plot(np.linspace(-3,3,2),maxPlumeHeight*np.ones((2,1)),color='black')
                plt.savefig(os.path.join(''.join(['./',foldername,'/heightTest',repr(print_counter).zfill(5),'.png'])),bbox_inches='tight')
                plt.close('all') 


###################################
#### Commenting for new method ####
###################################
#
#        # Sorts newZ and orders newX and newY in the same way.
#        zipped = zip(newZ,newX,newY)
#        zipped = sorted(zipped, key=lambda x: x[0])
#        newZ,newX,newY = zip(*zipped)
#
#        # Looking only at specific vertical slice.
#        y_slice = 0.0
#        tmp = np.asarray(newZ)
#        idx = (np.abs(tmp - y_slice)).argmin()
#        y_slice = tmp[idx]
#
##        newnewX = []
##        newnewY = []
##        newnewZ = []
##
##        for i in range(len(newY)):
##            if (newY[i] == y_slice):
##                newnewX.append(newX[i])
##                newnewY.append(newY[i])
##                newnewZ.append(newZ[i])
##  
##        newX = newnewX
##        newY = newnewY
##        newZ = newnewZ
#
#        coordsX1 = []
#        coordsX2 = []
#        coordsZ = []
#
#        count = 1
#        for k in range(len(newZ)-1):
#            if ( newZ[k] == newZ[k+1] ):
#                count = count + 1
#            else:
##                print(count)
#                sta = k-count+1
#                end = k+1
#                coordsX1.append(min(newX[sta:end]))
#                coordsX2.append(max(newX[sta:end]))
#                coordsZ.append(newZ[sta])
#                count = 1
#     
#        # deepcopy copies and removes any link to old file.  Otherwise reverse() will reverse both. 
#        coordsXr = copy.deepcopy(coordsX2) 
#        coordsXr.reverse()
#        coordsZ1 = copy.deepcopy(coordsZ)
#        coordsZ2 = copy.deepcopy(coordsZ)
#        coordsZ2.reverse()
#        coordsX = coordsX1 + coordsXr
#        coordsZ = coordsZ1 + coordsZ2
#
#        top_data.append(max(coordsZ1))
#        top_time.append(time)
#
#
#    ##############################################################
#    ######################### Plotting ###########################
#    ##############################################################
#
#        #fig = plt.figure(figsize=(12,11.6))
#        #plt.scatter(np.zeros(len(coordsZ1)),coordsZ1)
#        #axes = plt.gca()
#        #axes.set_xlim([-2,2])
#        #axes.set_ylim([0,6])
#
##        plt.savefig(os.path.join(''.join(['./Images/mesh',repr(file_num).zfill(5),'.png'])),\
##                bbox_inches='tight')
#
#        if (image_OnOff == 1):
#            fig = plt.figure(figsize=(12,6))   
# 
##            plt.scatter(coordsX1,coordsZ1,c='black')
##            plt.scatter(coordsX2,coordsZ1,c='black')
#
#            plt.plot(coordsX,coordsZ)
##            plt.scatter(np.ones(len(coordsZ1))*1.75,coordsZ1)
#
#            axes = plt.gca()
#            axes.set_xlim([-6,6])
#            axes.set_ylim([0,6])
#            plt.title(''.join(['time = %3.2f'%time]),fontsize=12)
#
#            plt.savefig(os.path.join(''.join(['./Images_outline/plumeOutline_',repr(file_num).zfill(5),'.png'])),\
#                bbox_inches='tight')
#
#            plt.close('all')
#
###################################
#### Commenting for new method ####
###################################

    # Closing files.
    f.close()
    g.close()

#    f = open('top_data.file','w')
#    for x in top_data:
#        f.write(str(x))
#        f.write("\n")
#    f.close()
#
#    f = open('top_time.file','w')
#    for x in top_time:
#        f.write(str(x))
#        f.write("\n")
#    f.close()

    for t in range(len((top_time))):
        if (top_time[t] > start_avg_at_time):
            start = t
            print(start)
            break

    top_mean = st.mean(top_data[start:-1])

    f = open('top_mean.file','w')
    f.write(str(top_mean))
    f.close()

# Plotting height vs. time
    fig = plt.figure(figsize=(12,6))

    plt.plot(top_time,top_data)
    plt.plot([top_time[0],top_time[-1]],[top_mean,top_mean])
    plt.plot([top_time[start],top_time[start]],[0,6])

    axes = plt.gca()
    axes.set_xlim([0,max(top_time)])
    axes.set_ylim([0,6])
    plt.xlabel('time',fontsize=12)
    plt.ylabel('height',fontsize=12)

    plt.savefig(os.path.join(''.join(['./',foldername,'/top_vs_time.png'])),bbox_inches='tight')
    plt.close('all')

    return
