#Link to path in which postProcess_lib is stored.
import sys
import os
print(' ')
# Counting number of files in directory.
files = os.listdir(".")
file_count = len([f for f in files if f[:20] == "plume_v1_production0"])
print('Number of files is ',file_count,'.')
# Number of files to skip each simulation.
jump = 500
print('Skip every ',jump,' files.')
# Computes number of calculations to perform.
to_calculate = int(round(float(file_count)/float(jump),0))
print('Performing ',to_calculate,' calculations.')

sys.path.insert(1,'/home/home01/scdrw/Python/scripts')
import compute_outline as co
#import make_videos as mv

# Chooses which postprocessing script to run.
switch = 1

def umbrellaOutline():
        co.umbrellaOutline('plume_v1_production',
        8,      # Order 
        3,      # Dimension
        1,      # Start file
        jump,      # Jump
        file_count,      # Final timestep
        1      # Image on/off
        )
        return

def plumeOutline():
        co.plumeOutline('plume_v1_production',
        8,      # Order 
        3,      # Dimension
        1,      # Start file
        jump,      # Jump
        file_count,      # Final timestep
        34,     # Number of elements in z-direction
        0.07,   # Cutoff value for s
        0,      # start averaging at this time
        0      # Image on/off
        )
        return

def makeVideo():
        mv.pseudoColour('plume_v1_production',
        8,      # Order 
        3,      # Dimension
        1,      # Start file
        jump,      # Jump
        file_count,      # Final timestep
        )
        return


def choose_function(argument):
        switcher = {
                0: umbrellaOutline,
                1: plumeOutline,
                2: makeVideo,
        }
        # Get the function from switcher dictionary
        func = switcher.get(argument)

        return func()

choose_function(switch)

