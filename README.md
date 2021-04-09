# STORM-Image-Analysis
Analysis codes to produce cluster images (DBSCAN) and calculation of the radius of gyration using ImageJ.

Run.py
April 2021
Author: Mathew Horrocks (mathew.horrocks@ed.ac.uk), adapted by Niamh Harper (s1752622@ed.ac.uk) for use with Python3.
Description: DBSCAN algorithim to generate clusters with a set minimum of localisations and distance between neighbouring localisations using a super-resolution generated image fitting results file.  All localisations that meet both eps and localisation threshold are written to an output file.  Individual clusters are located and plotted on a coordinate map, different clusters are given a different colour and a histogram is generated or how many locations are included in each cluster. A plotting prescision histogram is also generated.

Sizing.py
April 2021
Author: Mathew Horrocks (mathew.horrocks@ed.ac.uk), adapted by Niamh Harper (s1752622@ed.ac.uk) for use with Python3.
Description: Input super-resolution generated image fitting results file. Creates cluster analysis of input results and average x-, y-coordinates located. Generates localisation prescision as histogram. Output file of all cluster points with length and eccentricity measurements.  Output image file of average coordinates created.

myTag_measure_v1_2D.ijm
April 2021
Author: Davide Michieletto (davide.michieletto@gmail.com)
Description: Use with ImageJ macros. Image to analyse is opened and points to measure are selected.  Parameter of rectangle of interest are pre-set in the code.  Output file produces area of localisation of interest and eccentricity value.
