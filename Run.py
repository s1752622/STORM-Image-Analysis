import numpy as np
import pandas as pd
import os
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn import scatterplot

pathList = []

eps_threshold=2 #how close points are
minimum_locs_threshold=5 #min no. points to be ain a cluster

Filename='Fit_Results_t5.txt'   # This is the name of the SR file containing the localisations.

# Paths to analyse below:

pathList.append(r"/Users/niamhharper/Documents/Figures/DNA_4/new/t5")


for path in pathList:
    os.chdir(path)
    fit = pd.read_table(Filename, delimiter=',')
    fitcopy=pd.read_table(Filename, delimiter=',')
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
    fitall=fit.copy()
    toDelete = fit[ fit['Cluster'] == -1 ].index


    fit_truncated.drop(toDelete , inplace=True) # This deletes rows
    fit_truncated.drop(fit_truncated.columns[[0,7,16]],axis=1,inplace=True) # This drops the columns that aren't required for GDSCSMLM
    fit.drop(fit.columns[[0,7,16]],axis=1,inplace=True)


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
    Out_nc.write(fit.to_csv(sep = '\t',header=False,index=False))    # Write the columns that are required (without the non-clustered localisations)


    Out_nc.close() # Close the file.


    n_groups = n_clusters_+1 #total number of colours required is number of clusters +1 for the points not in clusters

        #choose colours to plot with. Black is first to represent points not in clusters. May need to add more if the number of clusters >25
    mycolours = ["#000000",'#1f78b4','#33a02c', '#fb9a99','#cab2d6','#6a3d9a',"#95a5a6","#34495e", '#e78ac3','#a6d854','#ffd92f','#e5c494','#b3b3b3','#a6cee3','#e31a1c','#fdbf6f','#ff7f00','#ffff99','#b15928', '#b2df8a',"#9b59b6", "#3498db", "#2ecc71",'#fc8d62','#8da0cb','#66c2a5']

    sns.scatterplot(x='X', y='Y',
                data = fitall,
                marker = ".",
                s = 2, #marker size
                edgecolor= None,
                hue= 'Cluster', #colour points depending on cluster number
                palette = sns.color_palette(mycolours, n_groups), #set colour palette to mycolours with the # of colours = # groups being plotted
                legend = False);

    plt.title('Cluster locations')
    plt.axis('equal')
    plt.savefig(path+'/'+'Clusters.pdf')
    plt.show()


    # Histogram of precisions
    precision=fit_truncated['Precision (nm)']
    plt.hist(precision, bins = 40,range=[0,40], rwidth=0.9,color='#607c8e')
    plt.xlabel('Precision (nm)')
    plt.ylabel('Number of Localisations')
    plt.title('Histogram of Precision')
    plt.savefig(path+'/'+'Precision.pdf')
    plt.show()



    # Calculate how many localisations per cluster
    clusters=labels.tolist()    # Need to convert the dataframe into a list- so that we can use the count() function.
    maximum=max(labels)+1       # This is the last cluster number- +1 as the loop goes to <end.
    cluster_contents=[]         # Make a list to store the number of clusters in

    for i in range(0,maximum):
        n=clusters.count(i)     # Count the number of times that the cluster number i is observed
        cluster_contents.append(n)  # Add to the list.
    c=sum(cluster_contents)
    print (c)
    plt.hist(cluster_contents, bins = 20,range=[0,200], rwidth=0.9,color='gray', edgecolor = 'dimgrey') # Plot a histogram.
    plt.xlabel('Precision (nm)')
    plt.xlabel('Localisations per cluster')
    plt.ylabel('Number of clusters')
    plt.title('Histogram of cluster size')
    plt.savefig(path+'/'+'Cluster_size.pdf')
    plt.show()
