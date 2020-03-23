# Post-processing in Python for Nek5000.
Used to post-process data from a Nek5000 simulation.  The folder ./scripts contains a number of files containing functions used to post-process Nek files, while the parent directory contains some outdated files and some used for converting binary data into a manipulatable (new word?) form.

The function readnek takes the Nek5000 output file (in binary) and reads it into a tensor.

The function reshapenek takes the output from readnek and uses the element map to map local elements back to their global positions on the grid.

The functions myPcolour and myPlot are simply used to plot a pseudo-colour plot and a line plot respectively.
