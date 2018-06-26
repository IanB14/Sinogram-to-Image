##
##  imutils.py.   Some image utilities, histograms, consistent histogram
##                plotting, etc.
##
##  imread     -- Load an image, with default conversion to greyscale.  Using
##                this takes care of the PIL Image interface for you.
##                Designed to replace Numpy's imread.
##  imshow     -- Display an image with useful defaults, like no smoothing,
##                no automatic range mapping, greyscale colourmap by default.
##                Designed to replace matplotlib.pyplot.imshow.
##  getChannel -- Extract one of the read, green or blue channels of a 24-bit
##                RGB colour image.  Returns an 8-bit image.
##
##
##  colhisto   -- Return a list of 3 histograms, 1 per channel, for an RGB 
##                24-bit colour image.
##
##  greyhisto  -- Return the histogram of an 8-bit, single-channel greyscale 
##                image.
##
##  chistplot  -- Display R,G and B histograms as a single figure, with the 
##                channels in the appropriate colours.
##
##  ghistplot  -- Display a histogram of a greyscale image.
##
##
##  brighten        -- Brighten an image by adding a constant to all pixels.  
##
##  contrastEnhance -- Contrast enhancement for 8-bit greyscale images.
##
##  autoContrast    -- Stretch an image to fill its entire range.
##
##  autoContrastSat -- Autocontrast with saturation.
##
##
##  equalize         -- Perform histogram equalization on an 8-bit greyscale 
##                      input image.  Simple but very slow implementation.
##                      Always use "equalize_v2".
##
##  equalize_v2      -- Same operation as "equalize" but a much more efficient
##                      implementation using fancy Numpy array indexing.
##                      Input image is 8-bit greyscale.
##
##  cl_equalize      -- Contrast-limited (i.e., slope-limited) histogram 
##                      equalization applied to an entire image.  You can 
##                      choose the maximum slope (defaults to 3.5) and whether
##                      or not to redistribute the excess (default True), plus
##                      some other, less important parameters.  Defaults are
##                      generally sensible.  Input image is 8-bit greyscale.
##
## subblock_equalize -- Equalization applied separately to N x M image 
##                      sub-blocks, the results of the separate equalizations
##                      then recombined into an output image.  No interpolation,
##                      not generally usefule, just for illustration. Input 
##                      image is 8-bit greyscale.
##
## adaptive_equalize -- Adaptive histogram equalization of an image using 
##                      N x M sub-blocks for calculation of the subblock 
##                      cumulative histograms, and bilinear interpolation 
##                      between block centres.  An implementation of an
##                      important practical technique.  Input image is 8-bit 
##                      greyscale.
##
## CLAHE             -- Contrast-limited adaptive histogram equalization of an
##                      image using N x M sub-blocks and a slope limit 
##                      (defaults to 3.5), with bilinear iinterpolation 
##                      between block centres.  An implementation of a very
##                      important practical technique.  Input image is 8-bit 
##                      greyscale.
##                     


import numpy as np
import matplotlib.pyplot as pyp
from PIL import Image


##
##  Utilities for loading, displaying and extracting colour channels from
##  images.
##

def imread(filename,greyscale=True):
    """Load an image, return as a Numpy array."""
    if greyscale:
        pil_im = Image.open(filename).convert('L')
    else:
        pil_im = Image.open(filename)
    return np.array(pil_im)


def imshow(im, autoscale=False,colourmap='gray', newfig=True, title=None):
    """Display an image, turning off autoscaling (unless explicitly required)
       and interpolation.
       
       (1) 8-bit greyscale images and 24-bit RGB are scaled in 0..255.
       (2) 0-1 binary images are scaled in 0..1.
       (3) Float images are scaled in 0.0..1.0 if their min values are >= 0
           and their max values <= 1.0
       (4) Float images are scaled in 0.0..255.0 if their min values are >= 0
           and their max values are > 1 and <= 255.0
       (5) Any image not covered by the above cases is autoscaled.  If 
           autoscaling is explicitly requested, it is always turned on.
           
       A new figure is created by default.  "newfig=False" turns off this
       behaviour.
       
       Interpolation is always off (unless the backend stops this).
    """
    if newfig: 
        if title != None: fig = pyp.figure(title)
        else: fig = pyp.figure()
    if autoscale:
        pyp.imshow(im,interpolation='nearest',cmap=colourmap)
    else:
        maxval = im.max()
        if im.dtype == 'uint8':        ## 8-bit greyscale or 24-bit RGB
            if maxval > 1: maxval = 255 
            pyp.imshow(im,interpolation='nearest',vmin=0,vmax=maxval,cmap=colourmap)
        elif im.dtype == 'float32' or im.dtype == 'float64':
            minval = im.min()
            if minval >= 0.0:
                if maxval <= 1.0:  ## Looks like 0..1 float greyscale
                    minval, maxval = 0.0, 1.0
                elif maxval <= 255.0: ## Looks like a float 0 .. 255 image.
                    minval, maxval = 0.0, 255.0 
            pyp.imshow(im,interpolation='nearest',vmin=minval,vmax=maxval,cmap=colourmap)
        else:
            pyp.imshow(im,interpolation='nearest',cmap=colourmap)
    pyp.axis('image')
    ## pyp.axis('off')
    pyp.show()
    ##return fig
    

def getChannel(im, channel='R'):
    """Pull the red (R), green (G) or blue (B) channel from an RGB image
       *im*.  Returns an RGB image with the other two channels zeroed."""
    rows,cols,channels = im.shape
    if channels < 3: 
        return im
    else:
        if channel == 'B':   chNum = 2
        elif channel == 'G': chNum = 1
        else:                chNum = 0
    chImage = np.zeros((rows,cols), dtype = im.dtype)
    chImage[:,:,chNum] = im[:,:,chNum]
    return chImage


##------------------------------------------------------------------------------
##
##  Histogram routines.
##
##

def colhisto(im):
    """Return a list of 3 histograms, 1 per channel, for an RGB 24-bit
       colour image."""
    assert im.shape[-1] == 3, "3-channel image assumed."
    assert im.dtype == 'uint8', "3-channel, 8-bits per channel image assumed."
    im_flat = im.reshape((-1,3))
    histos = []
    for chan in range(3):
        histos.append(np.bincount(im_flat[:,chan],minlength=256))
    return histos


def greyhisto(im):
    "Return the histogram of an 8-bit, single-channel greyscale image."
    assert len(im.shape) == 2, "Single-channel greyscale image required."
    assert im.dtype == 'uint8', "Single-channel, 8-bits per pixel image required."
    return np.bincount(im.ravel(),minlength=256)


def chistplot(hs,colour=None,ymax=None,newfig=True):
    """Plot R,G and B histograms (provided as a list of 3 256-element arrays)
       as a single figure, with the channels in the appropriate colours
       unless otherwise specified.
       
       If ymax is specifed, use this as the top of the y-range.
       If newfig is True (default), plot the histogram on a new figure."""
    if newfig: fig = pyp.figure()
    cols='rgb'
    for chan in range(3):
        if colour == None: c = cols[chan]
        else: c = colour
        pyp.plot(hs[chan],color=c,drawstyle='steps')
    if ymax == None: ymax = pyp.axis()[3]
    pyp.axis([0,256,0,ymax]) ## squeeze the x-axis
    pyp.show()
    ##return fig


def ghistplot(h,filled=True,barcolour='black',ymax=None,newfig=True):
    """Plot a histogram (provided as a 256-element array) of a greyscale image.
       
       If ymax is specifed, use this as the top of the y-range.
       If newfig is True (default), plot the histogram on a new figure.
       
       N.B. An image may be passed as the first parameter instead of the
       usual 256-element histogram.  If this is the case, a histogram of
       the image is generated and displayed."""    
    if newfig: fig = pyp.figure()
    if len(h.shape) != 1:  ## This must be an image, convert it to a histogram.
        h = greyhisto(h)
    if filled:
        pyp.bar(range(256),h,width=1,color=barcolour,linewidth=0)
    else:
        pyp.plot(h,color=barcolour,drawstyle='steps')
    if ymax == None: ymax = pyp.axis()[3]
    pyp.axis([0,256,0,ymax]) ## squeeze the x-axis
    pyp.show()
    ##return fig


##------------------------------------------------------------------------------
##
##  Simple image manipulations.
##
##


def brighten(im, brightenVal=10):
    """Brighten an image by adding a constant to all pixels.  Can be -ve.
       Clamped at 0 and 255."""
    assert len(im.shape) == 2, "1-channel image needed."
    assert im.dtype == 'uint8', "8-bit image needed."
    result = np.zeros(im.shape, dtype = 'uint8')
    
    np.clip(im + float(brightenVal), 0, 255, result)
    return result


def contrastEnhance(im, scaleFactor=1.2):
    "Contrast enhancement for 8-bit greyscale images."
    assert len(im.shape) == 2, "1-channel image needed."
    assert im.dtype == 'uint8', "8-bit image needed."
    result = np.zeros(im.shape, dtype = 'uint8')
    np.clip(im * float(scaleFactor), 0, 255, result)
    return result


def autoContrast(im):
    "Stretch an image to fill its entire range."
    assert len(im.shape) == 2, "1-channel image needed."
    assert im.dtype == 'uint8', "8-bit image needed."
    if im.min() > 0:
        im2 = im - im.min()
    else:
        im2 = im
    scaleFactor = 255.0 / im2.max()
    return contrastEnhance(im2, scaleFactor)
    

def autoContrastSat(im, sat=0.004):
    """Autocontrast with saturation. A fraction, sat, of pixels
       on each side of the histogram of the image will be saturated to
       0 and 255 respectively.  Default range end saturation is 0.4%."""
    size_im = im.shape[0] * im.shape[1]
    h_im = greyhisto(im)
    
    acc, i_lo, i_hi = 0, 0, 255
    sat_lo, sat_hi = int(sat*size_im), int((1.0-sat)*size_im)
    for i in range(len(h_im)):
        acc += h_im[i]
        if acc < sat_lo: i_lo = i
        if acc >= sat_hi:
            i_hi = i
            break
    print "Total image pixel count: %d" % size_im
    print "lower saturation value:  %d, at intensity: %d" % (sat_lo,i_lo)
    print "upper saturation value:  %d, at intensity: %d" % (sat_hi,i_hi)
    return contrastEnhance(im, 255.0/(i_hi - i_lo))
    

##------------------------------------------------------------------------------
##
##  Histogram equalization of images, and its various derivatives.
##
##
 
def equalize(im):
    """Perform histogram equalization on an 8-bit greyscale input
       image."""
    c_h = np.cumsum(greyhisto(im))  # Cumulative histogram in c_h
    size_im = c_h[-1]               # Total pixels in image is last entry
    R, C = im.shape                 # Get shape of source image
    im = im.ravel()                 # Flatten it for faster access (ravel is 
                                    # faster than flatten, which makes a copy.)
    im_eq = np.zeros(size_im,dtype = 'uint8')  ## Target image
    scale = 255.0 / size_im         # 'Resizing' constant to get output range
    for i in range(size_im):
        im_eq[i] = c_h[im[i]] * scale
    im_eq.resize((R,C))             # Reshape result in-place
    return im_eq


def equalize_v2(im):
    """Histogram equalization on an 8-bit greyscale input image, but
       this time using fancy Numpy array indexing to do the work."""
    R, C = im.shape
    im = im.ravel()
    c_h = np.cumsum(np.bincount(im,minlength=256))
    size_im = c_h[-1]
    assert size_im == R * C, "End of cumulative histogram must == R*C"
    ## All the work happens here.  Instead of an explicit loop, we use
    ## the flattened orignal image as an index into the cumulative histogram
    ## to generate a new, equalized (but flat) image, which we then scale
    ## appropriately, recast into bytes and reshape to the original image
    ## shape.
    return ((c_h[im] * 255) / size_im).astype('uint8').reshape((R,C))


def cl_equalize(im,maxslope=3.5,maxiter=10,redistribute=True,verbose=False):
    """Contrast-limited histogram equalization on an 8-bit greyscale input.
       maxslope is the maximum permitted slope in the normalised 0-1 
                cumulative histogram.  3.5 seems to be a good general choice,
                especially for CLAHE.
       maxiter is the maximum number of times to iterate the slope-limiting
                loop.
       redistribute is a boolean flag that controls whether or not to 
                redistribute the excess associated with a normalised histogram
                entry. Note that the slope-limiting loop will also exit if the
                amount to be distributed falls below 1e-6 per bin.  For CLHE
                redistribution doen't seem to make too much difference, but
                for CLAHE, it's advised.
       verbose  controls whether or not to output information about the 
                operation of the slope-limiting loop.
                """
    ## Generate the normalised slope-limited cumulative historgram and the
    ## slope-limited histogram (last parameter to makeCLchist = True).
    c_h = makeCLchist(im, maxslope, maxiter, redistribute, verbose)
    return c_h[im].astype('uint8').reshape(im.shape)


def subblock_equalize(im, r_blocks=3, c_blocks=4):
    """Histogram equalization applied to subblocks of an image.
       Very simple implementation, but note the use of np.concatenate
       to build the overall image from the list of equalized subimages."""
    R, C = im.shape
    dR = R / r_blocks
    dC = C / c_blocks
    assert dR * r_blocks == R, "r_blocks must divide R without remainder"
    assert dC * c_blocks == C, "c_blocks must divide C without remainder"
    subimages = []
    for r in range(r_blocks):
        for c in range(c_blocks):
            subimages.append(equalize_v2(im[r*dR:(r+1)*dR,c*dC:(c+1)*dC]))
    return np.concatenate([np.concatenate(subimages[i*c_blocks:(i+1)*c_blocks],
                                          axis = 1)
                        for i in range(r_blocks)], axis = 0)


def adaptive_equalize(im,rows,cols):
    """Adaptive histogram equalization of an image using rows x cols blocks
       for calculation of the subblock cumulative histograms, and bilinear
       interpolation between block centres. 
       
       Note.  No interpolation is performed at the corners of the image and
       linear interpolation only at the edges.
       
       The image must be evenly divisible (i.e. remainder = 0) by 2*rows
       and 2*cols."""
    assert (im.shape[0] / (2*rows)) * (2*rows) == im.shape[0]
    assert (im.shape[1] / (2*cols)) * (2*cols) == im.shape[1]
    ## Build the cumulative histograms for each of the rows * cols subblocks.
    chists = []
    dr, dc = im.shape[0]/rows, im.shape[1]/cols
    chists = []
    for r in range(rows):
        chistrow = []
        for c in range(cols):
            chistrow.append(np.cumsum(greyhisto(im[r*dr:(r+1)*dr,c*dc:(c+1)*dc])))
        chists.append(chistrow)
    return bilinear_interpolation(im, chists)
    

def CLAHE(im,rows,cols, maxslope=3.5, maxiter=10, redistribute=True, verbose=False):
    """Contrast Limited Adaptive Histogram Equalization of an image using 
       rows x cols blocks for calculation of the subblock contrst-limited 
       cumulative histograms, and bilinear interpolation between block centres. 
       
       Note.  No interpolation is performed at the corners of the image and
       linear interpolation only at the edges.
       
       The image must be evenly divisible (i.e. remainder = 0) by 2*rows
       and 2*cols."""
    assert (im.shape[0] / (2*rows)) * (2*rows) == im.shape[0]
    assert (im.shape[1] / (2*cols)) * (2*cols) == im.shape[1]
    ## Build the cumulative histograms for each of the rows * cols subblocks.
    chists = []
    dr, dc = im.shape[0]/rows, im.shape[1]/cols
    chists = []
    for r in range(rows):
        chistrow = []
        for c in range(cols):
            chistrow.append(makeCLchist(im[r*dr:(r+1)*dr,c*dc:(c+1)*dc], maxslope,\
                                        maxiter, redistribute, verbose))
        chists.append(chistrow)
    return bilinear_interpolation(im, chists)


##------------------------------------------------------------------------------
##
##  Support routines for adaptive equalization, CLAHE, etc.
##
##


def bilinear_interpolation(im, chists):
    """Bilinear interpolation using 2-d list of region histograms for rows x
       cols division of image.  Assumes image is 8-bit greyscale."""
    rows, cols = len(chists), len(chists[0])
    dr, dc = im.shape[0]/rows, im.shape[1]/cols
    ## Accumulate final equalized image in result.
    result = np.zeros(im.shape, im.dtype)
    rmin = dr / 2
    ## Now calculate the result image pixels for the central image region using
    ## bilinear interpolation between the "corner" cumulative histograms.
    for r in range(rows-1):
        rmax = rmin+dr
        cmin = dc / 2
        for c in range(cols-1):
            cmax = cmin+dc
            subim = im[rmin:rmax,cmin:cmax]
            result[rmin:rmax,cmin:cmax] = bilinear(subim,
                                                   chists[r][c], chists[r][c+1],
                                                   chists[r+1][c], chists[r+1][c+1])
            cmin = cmax
        rmin = rmax
        
    ## Left & right-hand edges, linear interpolation, implemented here as
    ## bilinear with only two unique cumulative histograms supplied.
    rmin = dr/2
    for r in range(rows-1):
        rmax = rmin+dr 
        subim = im[rmin:rmax,:dc/2]   # left-edge interpolation
        result[rmin:rmax,:dc/2]  = bilinear(subim,
                                            chists[r][0],chists[r][0],
                                            chists[r+1][0],chists[r+1][0])
        subim = im[rmin:rmax,-dc/2:]  # right-edge interpolation
        result[rmin:rmax,-dc/2:] = bilinear(subim,
                                            chists[r][-1],chists[r][-1],
                                            chists[r+1][-1],chists[r+1][-1])
        rmin = rmax

    # Top and bottom edges - linear interpolation.
    cmin = dc/2
    for c in range(cols-1):
        cmax = cmin+dc 
        subim = im[:dr/2,cmin:cmax]  # top edge interpolation
        result[:dr/2,cmin:cmax]  = bilinear(subim,
                                            chists[0][c],chists[0][c+1],
                                            chists[0][c],chists[0][c+1])
        subim = im[-dr/2:,cmin:cmax] # bottom edge interpolation
        result[-dr/2:,cmin:cmax] = bilinear(subim,
                                            chists[-1][c],chists[-1][c+1],
                                            chists[-1][c],chists[-1][c+1])
        cmin = cmax
    
    ## The four corners, no interpolation, just standard equalization.
    dr /= 2; dc /= 2
    result[:dr,:dc]   = 255 * chists[0][0][im[:dr,:dc]] / chists[0][0][-1]
    result[-dr:,:dc]  = 255 * chists[-1][0][im[-dr:,:dc]] / chists[-1][0][-1]
    result[:dr,-dc:]  = 255 * chists[0][-1][im[:dr,-dc:]] / chists[0][-1][-1]
    result[-dr:,-dc:] = 255 * chists[-1][-1][im[-dr:,-dc:]] / chists[-1][-1][-1]
    return result
    
    
def bilinear(im,htl,htr,hbl,hbr):
    """Perform bilinear interpolation on a (sub)image using 4 cumulative
       histograms: htl at the top-left corner of the image, htr at
       the top-right, hbl at the bottom left and hbr at the bottom right."""
    beta = np.linspace(0.0,1.0,im.shape[0])
    alpha = np.linspace(0.0,1.0,im.shape[1])
    r = np.outer(beta,alpha) * hbr[im] / hbr[-1]
    r += np.outer(beta,1-alpha) * hbl[im] / hbl[-1]
    r += np.outer(1-beta,alpha) * htr[im] / htr[-1]
    r += np.outer(1-beta,1-alpha) * htl[im] / htl[-1]
    return (255.0 * r).astype(im.dtype)



def makeCLhist(im, maxslope=3.5,maxiter=10,redistribute=False,verbose=True):
    """Return a contrast-limited histogram for an 8-bit greyscale
       input image.  Note that the output histogram is scaled in the range
       0..1.  Parameters are as for cl_equalize."""
    R, C = im.shape
    iteration = 1
    maxH = maxslope / 255.0             ## scale slope to account for spacing 0..255 in s
    if verbose: 
        print "Building slope-limited histogram, target slope =", maxslope,
        print "scaled as maxH =", maxH
    im = im.ravel()
    hs = np.bincount(im,minlength=256) / float(R * C)  ## histogram h(s) normalised
    ## Find indices of histogram entries exceeding the max (scaled) slope.
    mask = hs > maxH
    excess = sum(mask * (hs - maxH))
    while excess >= 1e-6 and iteration <= maxiter:
        excessCount = sum(mask)
        if verbose:
            print "Iteration", iteration
            print "    Total entries exceeding limit =", excessCount, "  Excess =", excess
        if redistribute:   # Clip excess entries at maxslope and redistribute excess to other bins.
            redist = excess / (256 - excessCount)
            if verbose: print "    Redistributing", redist, "to", (256-excessCount), "bins"
            newHs = (redist + hs) * (1-mask) + maxH * mask
            mask = newHs > maxH
            newExcess = sum(mask * (newHs - maxH))
            if verbose: print "    new excess =", newExcess
            if newExcess > excess:
                if verbose: print "    Excess is increasing, ", newExcess, "exiting."
                break
        else: # No redistribution of excess, just clip histogram at maxslope and force exit (by newExcess = 0).   
            newHs = hs * (1-mask) + maxH * mask
            newExcess = 0
        hs = newHs
        excess = newExcess
        iteration += 1
    return hs
    

def makeCLchist(im, maxslope=3.5,maxiter=10,redistribute=False,verbose=False):
    """Return a constrast-limited (i.e., slope-limited) Cumulative histogram
       for an 8-bit greyscale input image.  Note that the output cumulative
       histogram is scaled into the range 0 .. 255.0.  Parameters are as
       for cl_equalize."""
    hs = makeCLhist(im, maxslope, maxiter, redistribute, verbose)
    if verbose: print "Histogram sum =", hs.sum()
    chs = np.cumsum(hs)
    if verbose: print "Cumulative histogram max =", chs.max()    
    return 255* (chs /chs[-1])   # Note scaleing 0..1 then 0 ..255.


##-----------------------------------------------------------------------------
##
##  Some examples of using the routines, run if this is the main module, 
##  ignored if this is pulled in as a library by import.
##
##  Just CLAHE, imread and imshow for the moment (2017/1/1).
##

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import os
    files = os.listdir('.')
    names = ['pluto.png', 'pluto.jpg']
    found = False
    for name in names:
        if name in files:
            found = True
            break
    if found:
        pluto = imread(name)                          # imutils.imread.
        imshow(pluto, title='Original Pluto image')   # imutils.imshow.
        plutoCLAHE = CLAHE(pluto,4,4)                 # imutils.CLAHE.
        imshow(plutoCLAHE, title='Pluto CLAHE image, 4x4 subblocks, maxslope=3.5')
        plt.show()
