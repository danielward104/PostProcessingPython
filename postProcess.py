#Link to path in which postProcess_lib is stored.
import sys
import os
sys.path.insert(1,'/home/cserv1_a/soc_pg/scdrw/Documents/nbudocuments/PhD/SimNumerics/Python/postProcessingLib')
import postProcess_lib as pp

# Chooses which postprocessing script to run.
# 0 - PseudoColourPlotting
# 1 - integrateDomain
# 2 - integratePlume

switch = 0 

def outline():
	pp.computeOutline( 'plume_v1_production', 
	7,	# Order 
	3, 	# Dimension
	1, 	# Start file
	1, 	# Jump
	2, 	# Final timestep
	)
	return	
	
def choose_function(argument):
	switcher = {
		0: outline,
	}

	# Get the function from switcher dictionary
	func = switcher.get(argument)

	return func()

choose_function(switch)
