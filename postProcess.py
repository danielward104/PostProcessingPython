#Link to path in which postProcess_lib is stored.
import sys
import os
print(' ')
# Counting number of files in directory.
files = os.listdir(".")
file_count = len([f for f in files if f[:20] == "plume_v1_production0"])
print('Number of files is ',file_count,'.')
# Number of files to skip each simulation.
jump = 50
print('Skip every ',jump,' files.')
# Computes number of calculations to perform.
to_calculate = int(round(float(file_count)/float(jump),0))
print('Performing ',to_calculate,' calculations.')

# Number of elements in the z-direction
numelz = 34
print(' ')
print('Running with numelz = ',numelz,'.  Have you checked this is correct?')
print(' ')

# Insert path (for comp-pc6076 vs ARC vs VIPER).
#sys.path.insert(1,'/home/cserv1_a/soc_pg/scdrw/Documents/nbudocuments/PhD/SimNumerics/Python/postProcessingLib/scripts')      # Comp-pc6076
sys.path.insert(1,'/home/home01/scdrw/Python/scripts')          # ARC
#sys.path.insert(1,'/home/617122/Python/scripts')          # VIPER

# Chooses which postprocessing script to run.
switch = 1

def umbrellaOutline():
        import compute_outline as co
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
        import compute_outline as co
        co.plumeOutline('plume_v1_production',
        8,      # Order 
        3,      # Dimension
        1,      # Start file
        jump,      # Jump
        file_count,      # Final timestep
        numelz,     # Number of elements in z-direction
        0.03,   # Cutoff value for s
        0.5,      # start averaging at this time
        0      # Image on/off
        )
        return

def makeVideo():
        import make_videos as mv
        mv.pseudoColour('plume_v1_production',
        8,      # Order 
        3,      # Dimension
        1,      # Start file
        jump,      # Jump
        file_count,      # Final timestep
        numelz,     # Number of elements in z-direction
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

