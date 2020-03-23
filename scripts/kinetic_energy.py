import numpy as np
import numpy.fft as ft
import math
from array import array
import sys
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
# The following allows the script to work over SSH.  Check it still works when not SSH tunnelling.
plt.switch_backend('agg')
import os
from tempfile import TemporaryFile
from colorsys import hsv_to_rgb
import pickle
import copy
import statistics as st
import scipy.signal as sg
import scipy.interpolate as si

# Import user-defined modules (comp-pc6076 vs ARC).
sys.path.insert(1,'/home/cserv1_a/soc_pg/scdrw/Documents/nbudocuments/PhD/SimNumerics/Python/postProcessingLib/')
#sys.path.insert(1,'/home/home01/scdrw/Python')
import readingNek as rn
import mappings as mp
import plottingTools as pt
import generalTools as tools

# Print whole arrays instead of truncating.
np.set_printoptions(threshold=sys.maxsize)

def five_thirds_space( filename, start_file, jump, final_timestep, numel_z ):

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

        # Reads data files.
        data,time,istep,header,elmap,u_i,v_i,w_i,t_i,s_i = rn.readnek(''.join([filename, \
            '0.f',repr(k).zfill(5)]))


        ux = data[:,:,u_i]
        uy = data[:,:,v_i]
        uz = data[:,:,w_i]

#        fux = ft.fft2(ux,s=None)
#        print(fux)

        z_slice = 0.5

        Z = np.array(Z)
        nel,gll = np.shape(Z)
        order = int(round(gll**(1/3),0))

        grids = nel*gll

        inplanepoints = int(grids/(numel_z*order))

        Z_line = np.zeros(nel)
        idxs = np.zeros(nel)
        for el in range(0,nel):
            idx = (np.abs(Z[el,:] - z_slice)).argmin()
            idxs[el] = idx
            Z_line[el] = Z[el,idx]

        idx = (np.abs(Z_line - z_slice)).argmin() 
        z_slice = Z[idx,int(idxs[idx])]

        dim = 1 # Dimension of set over which to perform Fourier Transform

        if ( dim == 2 ):

            # Finding points in Z at the desired height (closest to z_slice).
            coords = np.zeros((inplanepoints,2))
            condition = 0
            test = 0.001
            while condition < 1:

                points_z = 0
                ell = 0
                odd = 0
                for el in range(0,nel):
                    ell_check = 1
                    for od in range(0,gll):
                        tester = abs(Z[el,od] - z_slice)
                        if ( tester < test ):
                            mapp = ell + odd
                            coords[mapp] = [el, od]
                            points_z += 1
                            if (ell_check > 0):
                                ell += 1
                                ell_check = 0
                            else:
                                odd += 1

                # Testing to ensure correct number of points are found for each slice.
                tester2 = int(points_z - inplanepoints)
                if ( tester2 < 0 ):
                    test = test*2
                elif ( tester2 > 0):
                    test = test/2
                else:
                    condition = 1

            coords = coords.astype(int)

            Xplane = X[coords[:,0],coords[:,1]]
            Yplane = Y[coords[:,0],coords[:,1]]
            uxplane = ux[coords[:,0],coords[:,1]]
            uyplane = uy[coords[:,0],coords[:,1]]

            lenXplane,lenYplane = len(Xplane),len(Yplane)

            ke = np.zeros(lenXplane)
            diff_fini = float('Inf')
            for i in range(lenXplane-1):
                ke[i] = uxplane[i]**2 + uyplane[i]**2 + uzline[i]**2

#                diff = abs(Xplane[i] - Xplane[i+1])
#                if ( diff < diff_fini ):
#                    diff_fini = diff
 
            # Interpolating to equally spaced points.
#            Length = int(round(12/diff_fini,0))

            XYplane = np.zeros((lenXplane,2))
            XYplane[:,0] = Xplane
            XYplane[:,1] = Yplane

            Length = lenXplane
            sample_frq = diff_fini
            Xdist = np.linspace(-6,6,Length)
            Ydist = np.linspace(-6,6,Length)
            kedist = si.griddata(XYplane,ke,(Xdist,Ydist))

            print(kedist)

            fig = plt.figure(figsize=(10,10))

            plt.imshow(kedist.T,extent=(-6,6,-6,6), origin='lower')

            plt.savefig(os.path.join(''.join(['test.png'])),bbox_inches='tight')
            plt.close('all')


########END if ( dim == 2 )

        if ( dim == 1 ):

            int_direction = 'vertical'

            if ( int_direction == 'horizontal' ):
                if ( k == 1 ):
                    print(' ')
                    print('Computing energy along horizontal line through zero.')

                # Finding points in Z at the desired height (closest to z_slice).
                coords = np.zeros((inplanepoints,2))
                condition = 0
                test = 0.001
                while condition < 1:
            
                    points_z = 0
                    ell = 0
                    odd = 0
                    for el in range(0,nel):
                        ell_check = 1
                        for od in range(0,gll):
                            tester = abs(Z[el,od] - z_slice)
                            if ( tester < test ):
                                mapp = ell + odd
                                coords[mapp] = [el, od]
                                points_z += 1
                                if (ell_check > 0):
                                    ell += 1
                                    ell_check = 0
                                else:
                                    odd += 1
        
                    # Testing to ensure correct number of points are found for each slice.
                    tester2 = int(points_z - inplanepoints)
                    if ( tester2 < 0 ):
                        test = test*2
                    elif ( tester2 > 0):
                        test = test/2
                    else:
                        condition = 1
        
                coords = coords.astype(int)
        
                # Finding line in z_slice.
    #            fig = plt.figure(figsize=(96,96))
        
                Xel = X[coords[:,0],coords[:,1]]
                Yel = Y[coords[:,0],coords[:,1]]
        
    #            plt.scatter(Xel,Yel,s=32,c='black')
        
                tester = 0.00001
                coords_line = np.zeros((len(Yel),2))
                count = 0
                for yz in range(0,len(Yel)):
                    if ( abs(Yel[yz]) < tester ):
                        coords_line[count,0] = coords[yz,0]
                        coords_line[count,1] = coords[yz,1]
                        count += 1
        
                coords_line = coords_line[0:count,:]
                coords_line = coords_line.astype(int)

                lenXline = int(count/2)
        
                # X,Y vals along a line in a z-slice.
                Xline = X[coords_line[:,0],coords_line[:,1]]
    #           Yline = np.zeros(lenXline)
                uxline = ux[coords_line[:,0],coords_line[:,1]]
                uyline = uy[coords_line[:,0],coords_line[:,1]]
                uzline = uz[coords_line[:,0],coords_line[:,1]]

                # Arranging x (and ux,uy,uz) in order of from smallest to largest.
                zipped = zip(Xline,uxline,uyline,uzline)
                zipped = sorted(zipped, key=lambda x: x[0])
                Xline,uxline,uyline,uzline = zip(*zipped)
        
                newXline = np.zeros(lenXline)
                newuxline = np.zeros(lenXline)
                newuyline = np.zeros(lenXline)
                newuzline = np.zeros(lenXline)
                for i in range(count):
                    if (i % 2) == 0:
                        j = int(i/2)
                        newXline[j] = Xline[i]
                        newuxline[j] = uxline[i]
                        newuyline[j] = uyline[i]
                        newuzline[j] = uzline[i]
        
                lenXline_old = lenXline
                lenXline = int(lenXline - lenXline/order + 1)
                newnewXline = np.zeros(lenXline)
                newnewuxline = np.zeros(lenXline)
                newnewuyline = np.zeros(lenXline)
                newnewuzline = np.zeros(lenXline)
                count2 = 0

                newnewXline[0] = newXline[0]
                newnewXline[-1] = newXline[-1]
                newnewuxline[0] = newuxline[0]
                newnewuxline[-1] = newuxline[-1]
                newnewuyline[0] = newuyline[0]
                newnewuyline[-1] = newuyline[-1]
                newnewuzline[0] = newuzline[0]
                newnewuzline[-1] = newuzline[-1]
        
                for i in range(1,lenXline_old-1):
                    if ( (i % order) != 0 ):
                        j = i - count2
                        newnewXline[j] = newXline[i]
                        newnewuxline[j] = newuxline[i]
                        newnewuyline[j] = newuyline[i]
                        newnewuzline[j] = newuzline[i]
                    else:
                        count2 += 1

                Xline = newnewXline
                Yline = np.zeros(lenXline)
                uxline = newnewuxline
                uyline = newnewuyline
                uzline = newnewuzline

                shorten = 1
                if ( shorten > 0 ):
                    sta = int(round(lenXline/2 - lenXline/3,0))
                    fin = int(round(lenXline/2 + lenXline/3,0))
                    Xline = Xline[sta:fin]
                    uxline = uxline[sta:fin]
                    uyline = uyline[sta:fin]
                    uzline = uzline[sta:fin]
                    lenXline = len(Xline)
                    Yline = np.zeros(lenXline)

                ke = np.zeros(lenXline)
                diff_fini = float('Inf')
                for i in range(lenXline-1):
                    ke[i] = uxline[i]**2 + uyline[i]**2 + uzline[i]**2

                    diff = abs(Xline[i] - Xline[i+1])
                    if ( diff < diff_fini ):
                        diff_fini = diff

                # Interpolating to equally spaced points.
                Length = int(round(12/diff_fini,0))
                sample_frq = diff_fini
                Xdist = np.linspace(-6,6,Length)
                kedist = np.interp(Xdist,Xline,ke)
                Ydist = np.zeros(Length)

                plt.scatter(Xdist,Ydist,s=60,c='blue')        
                plt.scatter(Xline,Yline,s=60,c='red')
                plt.savefig(os.path.join(''.join(['mesh_bw.png'])),bbox_inches='tight')
                plt.close('all')

                fig = plt.figure(figsize=(10,10))
                plt.plot(Xdist,kedist)
                plt.plot(Xline,ke)
                plt.savefig(os.path.join(''.join(['test.png'])),bbox_inches='tight')
                plt.close('all')


                # Computing Fourier Transform
                ke = np.fft.fft(kedist)
                freq = np.fft.fftfreq(Length)
                print('Note: need to scale frequencies!')

                Length = int(np.floor(Length/2) + 1)

                freq = freq[0:Length]/(Length*2)
                ke = ke[0:Length]
                ke[1:Length] = 2*ke[1:Length]

                # Plot 5/3 line
                a = math.exp(-24)
    #            x_53 = np.linspace(5*10**-6,10**-4,2)
                x_53 = np.linspace(5*10**-7,10**-5,2)
                y_53 = a*x_53**(-5/3)

                fig = plt.figure(figsize=(10,10))
                plt.loglog(freq,abs(ke)) 
                plt.loglog(x_53,y_53)
                plt.savefig(os.path.join(''.join(['ke_freq.png'])),bbox_inches='tight')
                plt.close('all')    


            elif ( int_direction == "vertical" ):
                if ( k == 1 ):
                    print(' ')
                    print('Computing energy along plume centre line in the vertical.')
                
                # Finding points at 0 (closest to zero).
                heightpoints = (numel_z*order)*4 # *4 because each surrounding element contributes.
                condition = 0
                test = 0.0001
                count = 0
                while condition < 1:

                    coords = []
                    points = 0
                    ell = 0
                    odd = 0
                    for el in range(0,nel):
                        ell_check = 1
                        for od in range(0,gll):
                            tester = X[el,od]**2 + Y[el,od]**2
                            if ( tester < test**2 ):
                                mapp = ell + odd
                                coords.append([el, od])
                                points += 1
                                if (ell_check > 0):
                                    ell += 1
                                    ell_check = 0
                                else:
                                    odd += 1

                    # Testing to ensure correct number of points are found for each slice.
                    tester2 = int(points - heightpoints)
                    if ( tester2 < 0 ):
                        test = test*2
                        count += 1
                    elif ( tester2 > 0):
                        test = test/2
                        count += 1
                    else:
                        condition = 1
       
                    # Print command to catch ever-iterating code. 
                    if ( count == 10 ):
                        print(' ')
                        print('Code is likely stuck in testing loop, check numel_z is correct in postprocess.py.')

                coords = np.array(coords)
                coords = coords.astype(int)

                Zline = Z[coords[:,0],coords[:,1]]
                uxline = ux[coords[:,0],coords[:,1]]
                uyline = uy[coords[:,0],coords[:,1]]
                uzline = uz[coords[:,0],coords[:,1]]

                # Arranging x (and ux,uy,uz) in order of from smallest to largest.
                zipped = zip(Zline,uxline,uyline,uzline)
                zipped = sorted(zipped, key=lambda x: x[0])
                Zline,uxline,uyline,uzline = zip(*zipped)

                heightpoints = int(heightpoints/4)

                newZline = np.zeros(heightpoints)
                newuxline = np.zeros(heightpoints)
                newuyline = np.zeros(heightpoints)
                newuzline = np.zeros(heightpoints)
                newheightpoints = heightpoints - numel_z + 1
                newnewZline = np.zeros(newheightpoints)
                newnewuxline = np.zeros(newheightpoints)
                newnewuyline = np.zeros(newheightpoints)
                newnewuzline = np.zeros(newheightpoints)
    
                for i in range(heightpoints):
                    newZline[i] = Zline[i*4]
                    newuxline[i] = uxline[i*4]
                    newuyline[i] = uyline[i*4]
                    newuzline[i] = uzline[i*4]

                newnewZline[0] = newZline[0]
                newnewZline[-1] = newZline[-1]
                newnewuxline[0] = newuxline[0]
                newnewuxline[-1] = newuxline[-1]
                newnewuyline[0] = newuyline[0]
                newnewuyline[-1] = newuyline[-1]
                newnewuzline[0] = newuzline[0]
                newnewuzline[-1] = newuzline[-1]

                count2 = 0
                for i in range(1,heightpoints-1):
                    if ( (i % order) != 0 ):
                        j = i - count2
                        newnewZline[j] = newZline[i]
                        newnewuxline[j] = newuxline[i]
                        newnewuyline[j] = newuyline[i]
                        newnewuzline[j] = newuzline[i]
                    else:
                        count2 += 1
                
                Zline = newnewZline
                uxline = newnewuxline
                uyline = newnewuyline
                uzline = newnewuzline

                heightpoints = newheightpoints

                shorten = 1
                if ( shorten > 0 ):
                    sta = 0
                    riseHeight = 4.35
                    fin = (np.abs(Zline - riseHeight)).argmin()
                    Zline = Zline[sta:fin]
                    uxline = uxline[sta:fin]
                    uyline = uyline[sta:fin]
                    uzline = uzline[sta:fin]
                    heightpoints = len(Zline)

                Yline = np.zeros(heightpoints)
    
                ke = np.zeros(heightpoints)
                diff_fini = float('Inf')
                for i in range(heightpoints-1):
                    ke[i] = uxline[i]**2 + uyline[i]**2 + uzline[i]**2

                    diff = abs(Zline[i] - Zline[i+1])
                    if ( diff < diff_fini ):
                        diff_fini = diff

                # Interpolating to equally spaced points.
                Length = int(round(6/diff_fini,0))
                sample_frq = diff_fini
                Zdist = np.linspace(0,6,Length)
                kedist = np.interp(Zdist,Zline,ke)
                Ydist = np.zeros(Length)

                fig = plt.figure(figsize=(10,20))
                plt.scatter(Ydist,Zdist,s=6,c='blue')
                plt.scatter(Yline,Zline,s=6,c='red')
                plt.savefig(os.path.join(''.join(['mesh_bw.png'])),bbox_inches='tight')
                plt.close('all')

                fig = plt.figure(figsize=(10,10))
                plt.plot(Zdist,kedist)
                plt.plot(Zline,ke)
                plt.savefig(os.path.join(''.join(['ke_vs_height.png'])),bbox_inches='tight')
                plt.close('all')
                

                # Computing Fourier Transform
                ke = np.fft.fft(kedist)
                freq = np.fft.fftfreq(Length)
                print('Note: need to scale frequencies!')

                Length = int(np.floor(Length/2) + 1)

                freq = freq[0:Length]/(Length*2)
                ke = ke[0:Length]
                ke[1:Length] = 2*ke[1:Length]

                # Plot 5/3 line
                a = math.exp(-11)
    #            x_53 = np.linspace(5*10**-6,10**-4,2)
                x_53 = np.linspace(10**-4,5*10**-5,2)
                y_53 = a*x_53**(-5/3)

                fig = plt.figure(figsize=(10,10))
                plt.loglog(freq,abs(ke))
                plt.loglog(x_53,y_53)
                plt.xlabel('Frequency')
                plt.ylabel('Kinetic energy')
                plt.savefig(os.path.join(''.join(['ke_freq.png'])),bbox_inches='tight')
                plt.close('all')

                

            else:
                print('Error in int_direction specification.')

#########END if ( dim == 1 )

        integrateinrealspace = 0

        if ( integrateinrealspace > 0 ):
 
            gllperel = order**2
            nelperz = int(len(coords)/gllperel)
            # Check
            if (nelperz != int(nel/numel_z) ):
                print('ERROR: Incorrect number of gll points found')
    
            fig = plt.figure(figsize=(6,6))
            numsubel = (order - 1)**2      
    
            tot_area = 0
            int_ux = 0

            for el in range(0,nelperz):
                i1 = el*gllperel
                i2 = (el+1)*gllperel - 1
                elcoords = coords[i1,0]
                gllcoords = coords[i1:i2,1]
 
                for i in range(0,len(gllcoords)):
                    Xel = X[elcoords,gllcoords[i]]
                    Yel = Y[elcoords,gllcoords[i]]

                    if (el == 1):
                        plt.scatter(Xel,Yel,s=32,c='red')
#                    elif (el == 2):
         

                for i in range(0,order-1):
                    for j in range(0,order-1):
                        if ( i*j < (order-2)**2 ):
                            # Calculating area of each sub-element. 
                            coord1 = i*order + j
                            coord2 = i*order + j + 1
                            coord3 = (i+1)*order + j
                            coord4 = (i+1)*order + j + 1
    
                            x1 = X[elcoords,gllcoords[coord1]]
                            x2 = X[elcoords,gllcoords[coord2]]
                            x3 = X[elcoords,gllcoords[coord3]]
                            x4 = X[elcoords,gllcoords[coord4]]
                            y1 = Y[elcoords,gllcoords[coord1]]
                            y2 = Y[elcoords,gllcoords[coord2]]
                            y3 = Y[elcoords,gllcoords[coord3]]
                            y4 = Y[elcoords,gllcoords[coord4]]
    
                            a = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
                            b = ((x4 - x2)**2 + (y4 - y2)**2)**0.5
                            c = ((x3 - x4)**2 + (y3 - y4)**2)**0.5
                            d = ((x1 - x3)**2 + (y1 - y3)**2)**0.5
    
                            p = ((x3 - x2)**2 + (y3 - y2)**2)**0.5
                            q = ((x4 - x1)**2 + (y4 - y1)**2)**0.5
                           
                            area = 0.25*( 4 * p**2 * q**2 - ( b**2 + d**2 - a**2 - c**2 )**2 )**0.5
                            tot_area = tot_area + area
        
                            ux1 = ux[elcoords,gllcoords[coord1]]
                            ux2 = ux[elcoords,gllcoords[coord2]]
                            ux3 = ux[elcoords,gllcoords[coord3]]
                            ux4 = ux[elcoords,gllcoords[coord4]]
                            uxsum = ux1 + ux2 + ux3 + ux4
                            int_ux =  int_ux + uxsum*area/4
    
    
    
#                            if ( el == 10000 ):
#                                if ( i == 0 ):
#                                    if ( j == 0 ):
#                                        plt.scatter(X[elcoords,gllcoords[coord1]],Y[elcoords,gllcoords[coord1]],s=16,c='blue')
#                                        plt.scatter(X[elcoords,gllcoords[coord2]],Y[elcoords,gllcoords[coord2]],s=16,c='black')
#                                        plt.scatter(X[elcoords,gllcoords[coord3]],Y[elcoords,gllcoords[coord3]],s=16,c='green')
#                                        plt.scatter(X[elcoords,gllcoords[coord4]],Y[elcoords,gllcoords[coord4]],s=16,c='yellow')
                print(' ')  
                print(tot_area)
                print(int_ux)
#            for i in range(0,len(gllcoords)):
#                Xel = X[elcoords,gllcoords[i]]
#                Yel = Y[elcoords,gllcoords[i]]
 
                 
               
     
                # Plotting proof that I can integrate over each element and sum. 
#                if (el == 1):
#                    plt.scatter(Xel,Yel,s=32,c='red')
#                elif (el == 2):
#                    plt.scatter(Xel,Yel,s=20,c='blue')
#                elif (el == 7):
#                    plt.scatter(Xel,Yel,s=10,c='green')
#                else:
#                    plt.scatter(Xel,Yel,s=5,c='black')
#
#                    if (i < 2):
#                        plt.scatter(Xel,Yel,s=16,c='black')
#
#                    if (i > order - 1) and (i < order + 2):
#                        plt.scatter(Xel,Yel,s=16,c='black')
#
#                    if (i > len(gllcoords)-4):
#                        plt.scatter(Xel,Yel,s=16,c='black')

                plt.savefig(os.path.join(''.join(['mesh_zoom.png'])),bbox_inches='tight')
                plt.close('all')
        
#########END if ( integrateinrealspace > 0 )






    return


def five_thirds_time( filename, start_file, jump, final_timestep, numel_z ):

    num_files = int((final_timestep - (start_file - 1))/jump)

    if (start_file == 1):
        range_vals = [x - (jump - 1) for x in np.array(range(1,num_files+1))*jump]
    else:
        range_vals = [x - (jump - 1) + start_file for x in np.array(range(1,num_files+1))*jump]


    # Reading in mesh data.
    data,time,istep,header,elmap,u_i,v_i,w_i,t_i,s_i = rn.readnek(''.join([filename,'0.f00001']))

    X = data[:,:,0]
    Y = data[:,:,1]
    Z = data[:,:,2]

    ux = data[:,:,u_i]
    uy = data[:,:,v_i]
    uz = data[:,:,w_i]

    Z = np.array(Z)
    nel,gll = np.shape(Z)
    order = int(round(gll**(1/3),0))
            
    # Finding points at 0 (closest to zero).
    heightpoints = (numel_z*order)*4 # *4 because each surrounding element contributes.
    condition = 0
    test = 0.0001
    count = 0
    while condition < 1:

        coords = []
        points = 0
        ell = 0
        odd = 0
        for el in range(0,nel):
            ell_check = 1
            for od in range(0,gll):
                tester = X[el,od]**2 + Y[el,od]**2
                if ( tester < test**2 ):
                    mapp = ell + odd
                    coords.append([el, od])
                    points += 1
                    if (ell_check > 0):
                        ell += 1
                        ell_check = 0
                    else:
                        odd += 1

        # Testing to ensure correct number of points are found for each slice.
        tester2 = int(points - heightpoints)
        if ( tester2 < 0 ):
            test = test*2
            count += 1
        elif ( tester2 > 0):
            test = test/2
            count += 1
        else:
            condition = 1

        # Print command to catch ever-iterating code. 
        if ( count == 10 ):
            print(' ')
            print('Code is likely stuck in testing loop, check numel_z is correct in postprocess.py.')

    coords = np.array(coords)
    coords = coords.astype(int)

    Zline = Z[coords[:,0],coords[:,1]]

    # Arranging coords in order of from smallest to largest z.
    zipped = zip(Zline,coords)
    zipped = sorted(zipped, key=lambda x: x[0])
    Zline,coords = zip(*zipped)

    newcoords = np.zeros((heightpoints,2))
    for i in range(0,heightpoints):
        newcoords[i,:] = coords[i]
    coords = newcoords.astype(int)

    heightpoints = int(heightpoints/4)

    newcoords = np.zeros((heightpoints,2))    
    newheightpoints = heightpoints - numel_z + 1
    newnewcoords = np.zeros((newheightpoints,2))

    for i in range(heightpoints):
        newcoords[i] = coords[i*4]

    newnewcoords[0,:] = newcoords[0,:]
    newnewcoords[-1,:] = newcoords[-1,:]

    count2 = 0
    for i in range(1,heightpoints-1):
        if ( (i % order) != 0 ):
            j = i - count2
            newnewcoords[j] = newcoords[i]
        else:
            count2 += 1
    
    coords = newnewcoords.astype(int)
    heightpoints = newheightpoints

    Zline = Z[coords[:,0],coords[:,1]]

   # Pick points to compute through time... 


    shorten = 1
    if ( shorten > 0 ):
        sta = int(round(heightpoints*0.05,0))
        fin = int(round(heightpoints*0.6,0))
        coords = coords[sta:fin]
        heightpoints = len(coords)

    total = 20
    inspectcoords = np.zeros((total-1,2))
    for i in range(0,total-1):
        j = int(round(heightpoints*i/total,0))
        inspectcoords[i] = coords[j,0],coords[j,1]

    coords = inspectcoords.astype(int)

#    ux = np.zeros((total-1,num_files))
#    uy = np.zeros((total-1,num_files))
#    uz = np.zeros((total-1,num_files))
    ke = np.zeros((total-1,num_files))
    time_record = np.zeros(num_files)

    for k in range_vals:

        # Outputs counter to terminal.
        if (start_file == 1):
            file_num = int((k-1)/jump + 1)
            files_remaining = int(num_files - file_num)
        else:
            file_num = int((k-1-start_file)/jump + 1)
            files_remaining = int(num_files - file_num)

        sys.stdout.write("\r")
        sys.stdout.write("Files remaining: {:3d}".format(files_remaining))
        sys.stdout.flush()

        # Reads data files.
        data,time,istep,header,elmap,u_i,v_i,w_i,t_i,s_i = rn.readnek(''.join([filename, \
            '0.f',repr(k).zfill(5)]))

#        ux[:,file_num-1] = data[coords[:,0],coords[:,1],u_i]
#        uy[:,file_num-1] = data[coords[:,0],coords[:,1],v_i]
#        uz[:,file_num-1] = data[coords[:,0],coords[:,1],w_i]

        ux = data[coords[:,0],coords[:,1],u_i]
        uy = data[coords[:,0],coords[:,1],v_i]
        uz = data[coords[:,0],coords[:,1],w_i]

        ke[:,file_num-1] =  ux**2 + uy**2 + uz**2

        time_record[file_num-1] = time

    print(' ')
    print('Computing Fourier Transform.')

    dt = np.zeros(num_files-1)
    sample_frq = float('Inf')
    ke_avg = np.zeros(num_files)

    for i in range(num_files-1):
        dt[i] = time_record[i+1] - time_record[i]
        diff = dt[i]
        if ( diff < sample_frq ):
            sample_frq = diff

    sample_frq = sample_frq/1000

    for i in range(total-1):
        ke_avg += ke[i,:]

    ke_avg = ke_avg/num_files
   
    # Interpolating to equally spaced points.
    Length = int(round((time_record[-1]-time_record[0])/sample_frq,0))
    Tdist = np.linspace(time_record[0],time_record[-1],Length)
    kedist = np.interp(Tdist,time_record,ke_avg)
    Ydist = np.zeros(Length)

    fig = plt.figure(figsize=(20,10))
    plt.scatter(Tdist,np.zeros(Length),s=6,c='blue')
    plt.scatter(Tdist,np.ones(Length)*0.0002,s=6,c='blue')
    plt.scatter(time_record,np.zeros(num_files),s=6,c='red')
    plt.scatter(time_record,np.ones(num_files)*-0.0002,s=6,c='red')
    plt.savefig(os.path.join(''.join(['time_mesh.png'])),bbox_inches='tight')
    plt.close('all')

    fig = plt.figure(figsize=(10,10))
    plt.plot(time_record,ke_avg,c='red')
    plt.plot(Tdist,kedist,c='blue')
    plt.savefig(os.path.join(''.join(['interp_vs_orginal.png'])),bbox_inches='tight')
    plt.close('all')

    # Computing Fourier Transform
    ke_ft = np.fft.fft(kedist)
    freq = np.fft.fftfreq(Length)
    print('Note: need to scale frequencies!')

    Length = int(np.floor(Length/2) + 1)

    freq = freq[0:Length]/(Length*2)
    ke_ft = abs(ke_ft[0:Length])
    ke_ft[1:Length] = 2*ke_ft[1:Length]

    # Plot 5/3 line
    a = math.exp(-11)
#            x_53 = np.linspace(5*10**-6,10**-4,2)
    x_53 = np.linspace(10**-4,5*10**-5,2)
    y_53 = a*x_53**(-5/3)

    fig = plt.figure(figsize=(15,15))
    plt.loglog(freq,ke_ft)
    plt.loglog(x_53,y_53)
    plt.xlabel('Frequency')
    plt.ylabel('Kinetic energy')
    plt.savefig(os.path.join(''.join(['ke_freq.png'])),bbox_inches='tight')
    plt.close('all')


    return


