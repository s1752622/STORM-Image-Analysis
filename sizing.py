import numpy as np
import pandas as pd
import os
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import skimage as sk
import scipy
import scipy.ndimage
import scipy.stats
import scipy.misc
import skimage.measure
import skimage.morphology
import copy
#from scipy.misc import imsave
import imageio

# Thresholds that need changing are here
pathList = []

eps_threshold=1.0
minimum_locs_threshold=10.0
skel_threshold=1
skel_gauss_sigma=2
skel_cluster_size=5
PIXEL_SIZE=103.0

Filename='Fit_Results_hh.txt'   # This is the name of the SR file containing the localisations.

# Paths to analyse below:

pathList.append(r"/Users/niamhharper/Documents/Figures/skeleton")


def init(path):

    print ('RUNNING')
    os.chdir(path)
    fit = pd.read_table(Filename,delimiter=',')
    fitcopy=pd.read_table(Filename,delimiter=',')
    F = np.array(list(zip(fit['X'],fit['Y'])))
    try:
        db = DBSCAN(eps=eps_threshold, min_samples=minimum_locs_threshold).fit(F)
        labels = db.labels_
        n_clusters_ = len(set(labels)) - (1 if-1 in labels else 0)  # This is to calculate the number of clusters.
        print('Estimated number of clusters: %d' % n_clusters_)

        fit['Cluster'] = labels
        fit.to_csv(path + '/' + 'Results_clustered_eps_'+str(eps_threshold)+'_locs_'+str(minimum_locs_threshold)+'.csv', sep = '\t')
    except ValueError:
        pass


    # This is to delete the rows which are not part of a cluster:
    fit_truncated=fit.copy()
    fit_all=fit.copy()

    toDelete = fit_all[ fit_all['Cluster'] == -1 ].index


    fit_truncated.drop(toDelete , inplace=True) # This deletes rows
    fit_truncated.drop(fit_truncated.columns[[0,7,16]],axis=1,inplace=True) # This drops the columns that aren't required for GDSCSMLM
    fit_all.drop(fit_all.columns[[0,7,16]],axis=1,inplace=True)


    Out=open(path+'/'+'Clustered_FitResults_eps_'+str(eps_threshold)+'_locs_'+str(minimum_locs_threshold)+'.txt','w')   # Open file for writing to.


    # Write the header of the file
    Out.write("""#Localisation Results File
#FileVersion Text.D0.E0.V2
#Name Clustered (LSE)
#Source <gdsc.smlm.ij.IJImageSource><name>Clustered</name><width>512</width><height>512</height><frames>200</frames><singleFrame>0</singleFrame><extraFrames>0</extraFrames><path></path></gdsc.smlm.ij.IJImageSource>
#Bounds x0 y0 w512 h512
#Calibration <gdsc.smlm.results.Calibration><nmPerPixel>103.0</nmPerPixel><gain>55.5</gain><exposureTime>50.0</exposureTime><readNoise>0.0</readNoise><bias>0.0</bias><emCCD>false</emCCD><amplification>0.0</amplification></gdsc.smlm.results.Calibration>
#Configuration <gdsc.smlm.engine.FitEngineConfiguration><fitConfiguration><fitCriteria>LEAST_SQUARED_ERROR</fitCriteria><delta>1.0E-4</delta><initialAngle>0.0</initialAngle><initialSD0>2.0</initialSD0><initialSD1>2.0</initialSD1><computeDeviations>false</computeDeviations><fitSolver>LVM</fitSolver><minIterations>0</minIterations><maxIterations>20</maxIterations><significantDigits>5</significantDigits><fitFunction>CIRCULAR</fitFunction><flags>20</flags><backgroundFitting>true</backgroundFitting><notSignalFitting>false</notSignalFitting><coordinateShift>4.0</coordinateShift><shiftFactor>2.0</shiftFactor><fitRegion>0</fitRegion><coordinateOffset>0.5</coordinateOffset><signalThreshold>0.0</signalThreshold><signalStrength>30.0</signalStrength><minPhotons>0.0</minPhotons><precisionThreshold>400.0</precisionThreshold><precisionUsingBackground>true</precisionUsingBackground><nmPerPixel>117.0</nmPerPixel><gain>55.5</gain><emCCD>false</emCCD><modelCamera>false</modelCamera><noise>0.0</noise><minWidthFactor>0.5</minWidthFactor><widthFactor>1.01</widthFactor><fitValidation>true</fitValidation><lambda>10.0</lambda><computeResiduals>false</computeResiduals><duplicateDistance>0.5</duplicateDistance><bias>0.0</bias><readNoise>0.0</readNoise><amplification>0.0</amplification><maxFunctionEvaluations>2000</maxFunctionEvaluations><searchMethod>POWELL_BOUNDED</searchMethod><gradientLineMinimisation>false</gradientLineMinimisation><relativeThreshold>1.0E-6</relativeThreshold><absoluteThreshold>1.0E-16</absoluteThreshold></fitConfiguration><search>2.5</search><border>1.0</border><fitting>3.0</fitting><failuresLimit>10</failuresLimit><includeNeighbours>true</includeNeighbours><neighbourHeightThreshold>0.3</neighbourHeightThreshold><residualsThreshold>1.0</residualsThreshold><noiseMethod>QUICK_RESIDUALS_LEAST_MEAN_OF_SQUARES</noiseMethod><dataFilterType>SINGLE</dataFilterType><smooth><double>0.5</double></smooth><dataFilter><gdsc.smlm.engine.DataFilter>MEAN</gdsc.smlm.engine.DataFilter></dataFilter></gdsc.smlm.engine.FitEngineConfiguration>
#Frame	origX	origY	origValue	Error	Noise	Background	Signal	Angle	X	Y	X SD	Y SD	Precision

    """)
    Out.write(fit_truncated.to_csv(sep = '\t',header=False,index=False))    # Write the columns that are required (without the non-clustered localisations)


    Out.close() # Close the file.


    Out_nc=open(path+'/'+'FitResults_withheader.txt','w')   # Open file for writing to.
        # Write the header of the file
    Out_nc.write("""#Localisation Results File
#FileVersion Text.D0.E0.V2
#Name Clustered (LSE)
#Source <gdsc.smlm.ij.IJImageSource><name>Clustered</name><width>512</width><height>512</height><frames>200</frames><singleFrame>0</singleFrame><extraFrames>0</extraFrames><path></path></gdsc.smlm.ij.IJImageSource>
#Bounds x0 y0 w512 h512
#Calibration <gdsc.smlm.results.Calibration><nmPerPixel>103.0</nmPerPixel><gain>55.5</gain><exposureTime>50.0</exposureTime><readNoise>0.0</readNoise><bias>0.0</bias><emCCD>false</emCCD><amplification>0.0</amplification></gdsc.smlm.results.Calibration>
#Configuration <gdsc.smlm.engine.FitEngineConfiguration><fitConfiguration><fitCriteria>LEAST_SQUARED_ERROR</fitCriteria><delta>1.0E-4</delta><initialAngle>0.0</initialAngle><initialSD0>2.0</initialSD0><initialSD1>2.0</initialSD1><computeDeviations>false</computeDeviations><fitSolver>LVM</fitSolver><minIterations>0</minIterations><maxIterations>20</maxIterations><significantDigits>5</significantDigits><fitFunction>CIRCULAR</fitFunction><flags>20</flags><backgroundFitting>true</backgroundFitting><notSignalFitting>false</notSignalFitting><coordinateShift>4.0</coordinateShift><shiftFactor>2.0</shiftFactor><fitRegion>0</fitRegion><coordinateOffset>0.5</coordinateOffset><signalThreshold>0.0</signalThreshold><signalStrength>30.0</signalStrength><minPhotons>0.0</minPhotons><precisionThreshold>400.0</precisionThreshold><precisionUsingBackground>true</precisionUsingBackground><nmPerPixel>117.0</nmPerPixel><gain>55.5</gain><emCCD>false</emCCD><modelCamera>false</modelCamera><noise>0.0</noise><minWidthFactor>0.5</minWidthFactor><widthFactor>1.01</widthFactor><fitValidation>true</fitValidation><lambda>10.0</lambda><computeResiduals>false</computeResiduals><duplicateDistance>0.5</duplicateDistance><bias>0.0</bias><readNoise>0.0</readNoise><amplification>0.0</amplification><maxFunctionEvaluations>2000</maxFunctionEvaluations><searchMethod>POWELL_BOUNDED</searchMethod><gradientLineMinimisation>false</gradientLineMinimisation><relativeThreshold>1.0E-6</relativeThreshold><absoluteThreshold>1.0E-16</absoluteThreshold></fitConfiguration><search>2.5</search><border>1.0</border><fitting>3.0</fitting><failuresLimit>10</failuresLimit><includeNeighbours>true</includeNeighbours><neighbourHeightThreshold>0.3</neighbourHeightThreshold><residualsThreshold>1.0</residualsThreshold><noiseMethod>QUICK_RESIDUALS_LEAST_MEAN_OF_SQUARES</noiseMethod><dataFilterType>SINGLE</dataFilterType><smooth><double>0.5</double></smooth><dataFilter><gdsc.smlm.engine.DataFilter>MEAN</gdsc.smlm.engine.DataFilter></dataFilter></gdsc.smlm.engine.FitEngineConfiguration>
#Frame	origX	origY	origValue	Error	Noise	Background	Signal	Angle	X	Y	X SD	Y SD	Precision

    """)
    Out_nc.write(fit_all.to_csv(sep = '\t',header=False,index=False))    # Write the columns that are required (without the non-clustered localisations)


    Out_nc.close() # Close the file.


    # Histogram of precisions
    precision=fit_truncated['Precision (nm)']
    plt.hist(precision, bins = 40,range=[0,40], rwidth=0.9,color='#607c8e')
    plt.xlabel('Precision (nm)')
    plt.ylabel('Number of Localisations')
    plt.savefig(path+'/'+'Precision.pdf')
    plt.show()

    # Calculate how many localisations per cluster
    clusters=labels.tolist()    # Need to convert the dataframe into a list- so that we can use the count() function.
    maximum=max(labels)+1       # This is the last cluster number- +1 as the loop goes to <end.
    cluster_contents=[]         # Make a list to store the number of clusters in

    for i in range(0,maximum):
        n=clusters.count(i)     # Count the number of times that the cluster number i is observed
        cluster_contents.append(n)  # Add to the list.


    plt.hist(cluster_contents, bins = 20,range=[0,200], rwidth=0.9,color='#607c8e') # Plot a histogram.
    plt.xlabel('Localisations per cluster')
    plt.ylabel('Number of clusters')
    plt.savefig(path+'/'+'Localisations.pdf')
    plt.show()
    coords_and_clusters=pd.DataFrame(zip(fit.Cluster,fit.X,fit.Y))

    skele, clusters,meanx,meany = skeletonize(arr=coords_and_clusters.to_numpy(),directory=path,threshold=skel_threshold,gauss_sigma=skel_gauss_sigma,cluster_size=skel_cluster_size)
    if skele.shape == (4096, 4096):
        final_counts, ecc = count(skele, clusters)



        save_data(final_counts, ecc, meanx, meany, path)


    plt.hist(final_counts, bins = 20,range=[0,400], rwidth=0.9,color='#607c8e')
    plt.xlabel('Length (nm)')
    plt.ylabel('Number of Aggregates')
    plt.savefig(path+'/'+'Lengths.pdf')
    plt.show()
    return final_counts,meanx,meany

# Skeletonisation script:

def skeletonize(arr,directory,threshold, gauss_sigma, cluster_size):
    print ('Skeletonizing...')

    xcoord=[]
    ycoord=[]
    labels = {}

    labelled = np.zeros((512*8, 512*8))#, dtype='uint16')
    try:
        max_val = int(np.amax(arr[:,0]+1))
        print (max_val)
    except IndexError:
        print ('File contains no clusters!\n')
        return np.zeros((10, 10)), np.zeros((10, 10))

    for label in range(1, max_val):
        a = arr[np.where(arr[:,0] == label)]
        a = np.delete(a, np.s_[0], axis=1)
        a = a*8
        a = a.astype('int')

        for xy in a:
            x = xy[0]
            y = xy[1]
            labelled[x][y] = label
            labels[(x, y)] = label

        meanx=np.mean(xy[0])    # This is to calculate the mid x,y positions.
        meany=np.mean(xy[1])

        xcoord.append(meanx)
        ycoord.append(meany)
        #print 'The mean x is %g and the mean y is %g'%(meanx,meany)

    clusters = labelled.astype('int')
    clusters2 = copy.deepcopy(clusters)
    clusters3 = np.where(clusters2>0, 1, 0)


    skele1 = sk.morphology.skeletonize_3d(clusters3)
    skele1 = skele1*100
    skele1_blurred = scipy.ndimage.filters.gaussian_filter(skele1, gauss_sigma)
    skele1_blurred = skele1_blurred.astype(bool)
    skele2 = sk.morphology.skeletonize_3d(skele1_blurred)
    skele2 = skele2.astype('int8')
    sp = os.path.join(directory, ('skele.tif'))
    #imsave(sp, skele2)


    # >>> img = imageio.imread('img.jpg')
    # >>> img.dtype
    # dtype('uint8')
    # imageio.imwrite('skele.tif', sp)
    imageio.imsave(sp, skele2)
    # img_read = imageio.imread('skele.tif')
    # img_read.dtype
    # dtype('int16')



    ii = zip(*np.where(skele2 > 0))
    for i in ii:
        x, y = i
        try:
            skele2[i] = labels[i]
        except KeyError:
            skele2[i] = 0

    return skele2, clusters,xcoord,ycoord


def count(skele, clusters):
    print ('Counting...')
    nm_lengths = []
    ecc = []

    max_val = np.amax(skele)

    props = skimage.measure.regionprops(clusters)
    for val in range(1, max_val+1):
        final_count = 0
        ii = zip(*np.where(skele==val))
        counted = []
        counts = []
        for i in ii:
            count = 0
            x, y = i
            neighbours = [(x-1, y-1), (x-1, y), (x-1, y+1), (x, y-1), (x, y+1), (x+1, y-1), (x+1, y), (x+1, y+1)]
            for x in range(1, 9):
                if skele[neighbours[x-1]] == val:
                    if (x not in [2,4,5,7]) and (neighbours[x-1] not in counted):
                        final_count += np.sqrt(2)
                    else:
                        final_count += 1
                else:
                    count += 1
            if count == 8:
                final_count += 1
            counts.append(count)
            counted.append(i)

        try:
            ecc.append(props[val-1]['eccentricity'])
        except IndexError:
            pass
        if final_count == 0:
            final_count = 1
        nm_lengths.append(final_count/8.0*PIXEL_SIZE)

    return nm_lengths, ecc



def save_data(final_counts, ecc, meanx, meany, d):
    print ('Saving...\n\n'),
    if not os.path.exists(d):
        os.makedirs(d)
    fi = os.path.join(d, 'lengths_and_eccentricity_coords.txt')
    with open(fi, 'w') as f:
        for x1,x2,x3,x4 in zip(final_counts,ecc,meanx,meany):
            f.write(str(x1) + '\t' + str(x2) + '\t' + str(x3) + '\t' + str(x4) + '\n')

for path in pathList:
    init(path)
