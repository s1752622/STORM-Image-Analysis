name = getTitle();
dir=getDirectory("Choose Output Directory")
f=File.open(dir + name + ".txt");

//run("Gaussian Blur...", "sigma=10");
//run("Enhance Contrast", "saturated=0.35");
run("Apply LUT");

//////////////////////////////////////////////////////////////////
///  This is to remove inch/cm units and go back to pixels    ////
//////////////////////////////////////////////////////////////////
run("Set Scale...", "distance=0");

setTool("multipoint");
waitForUser("Please select center points for all areas of interest. Click OK when done")
run("Clear Results");
run("Measure");
 for (j=0; j<nResults; j++) {
 px = getResult("X",j);
 py = getResult("Y",j);
//the coordinates index from the top left, like a 2D array
 makeRectangle(px-40, py-40, 60, 60); 
 roiManager("Add");
 run("Select None");
}
//NOW I HAVE A LIST OF ROIs TO ANALYSE

for (m = 0; m < roiManager("count"); m++){
selectWindow(name);
run("Clear Results");

roiManager("Select", m);
run("Duplicate...", "title=roi");
roi = getImageID();

//run("Split Channels");

//////////////////////////////////////////////////////////////////////////////////////
/// IMPORTANT!
///  ASSUMES Analyse->Set Measurements -> Area and Shape Descr are Ticked!
//////////////////////////////////////////////////////////////////////////////////////

//RED CHANNEL
selectWindow("roi");

///////////////////////////////////
// THRESHOLD THE IMAGE 	//////////
//////////////////////////////////
// Get 10% brightest pixels from ROI

h = getHeight(); 						// x size of the image
w = getWidth();  						// y size of the image
n = w*h; 								// Total number of pixels
nBins = 65536; 							// Histogram bins
values = newArray(nBins);
getHistogram(values, counts, nBins);	// Create the histogram
cum = newArray(nBins);					// Cumulative histogram
cum[0] = counts[0]; 					// Start is just the first bin
cdf = newArray(nBins);					// Percentage of emitters
cdf[0] = cum[0]/n;						// Percentage of emitter in the first bin
for (k = 1; k < nBins; k++){ 			// Cycler over the bins
        cum[k] = counts[k] + cum[k-1]; 	// Creation of cumulative histogram
        cdf[k] = cum[k]/n; 				// Percentage of emitters
        //print(k,counts[k],values[k],cum[k],cdf[k]);
        if (cdf[k]*100 > 98.0)			// If percentage is greater than 98% we go out from the cycler
        {
        //one_percent=values[k];
        two_percent=k;
        break;
        }
}

// Set threshold to 2% brightest pixels (need to convert to mask)
//setThreshold(two_percent, 1000000000000000000000000000000.0000);
//////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////
// AUTO THRESHOLD
///////////////////////////////////////////////////////////////////////////
setAutoThreshold("RenyiEntropy dark");

///////////////////////////////////////////////////////////////////////////
// MASK AND MEASURE
///////////////////////////////////////////////////////////////////////////
setOption("BlackBackground", true);
run("Convert to Mask");

run("Analyze Particles...", "size=2-Infinity show=Outlines display");
redx = getResult("X",0);
redy = getResult("Y",0);
Area = getResult("Area",0);
AR = getResult("Circ.",0); //aspect ratio

print(Area,AR);

//run("Clear Results");


//PRINT TO FILE
print(f, m + "," + Area + "," + AR);

//CLOSE
selectWindow("roi");
close();
//selectWindow("Drawing of roi");
//close();

}
File.close(f);
