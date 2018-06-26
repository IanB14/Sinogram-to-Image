"Imports"
import numpy as np 
import imutils
from skimage.transform import rotate ## Image rotation routine
import scipy.fftpack as fft          ## Fast Fourier Transform
import scipy.misc                    ## Contains a package to save numpy arrays as .PNG



## Methods    


"Radon transform method - turns an image into a sinogram"
def radon(image, steps):        
    #Build the Radon Transform using 'steps' projections of 'image'. 
    projections = []        ## Accumulate projections in a list.
    dTheta = -180.0 / steps ## Angle increment for rotations.
    
    for i in range(steps):
        projections.append(rotate(image, i*dTheta).sum(axis=0))
    
    return np.vstack(projections) # Return the projections as a sinogram
    


"Translate the sinogram to the frequency domain using Fourier Transform"
def fft_translate(projs):
    #Build 1-d FFTs of an array of projections, each projection 1 row of the array.
    return fft.rfft(projs, axis=1)



"Filter the projections using a ramp filter"
def ramp_filter(ffts):
    #Ramp filter a 2-d array of 1-d FFTs (1-d FFTs along the rows).
    ramp = np.floor(np.arange(0.5, ffts.shape[1]//2 + 0.1, 0.5))
    return ffts * ramp

"Return to the spatial domain using inverse Fourier Transform"
def inverse_fft_translate(operator):
    return fft.irfft(operator, axis=1)



"Reconstruct the image by back projecting the filtered projections (UNFINISHED)"
def back_project(operator):
    laminogram = np.zeros((operator.shape[1],operator.shape[1]))
    dTheta = 180.0 / operator.shape[0]
    for i in range(operator.shape[0]):
        temp = np.tile(operator[i],(operator.shape[1],1))
        temp = rotate(temp, dTheta*i)
        laminogram += temp
    return laminogram



## Statements



"Import the image as a numpy array and display the original sinogram image"
print("Original Sinogram")
sinogram = imutils.imread('sinogram.png')
imutils.imshow(sinogram)
scipy.misc.imsave('originalSinogramImage.png', sinogram)



"Attempt to reconstruct the image directly from the sinogram without any kind of filtering"
print("Reconstruction with no filtering")
unfiltered_reconstruction = back_project(sinogram)
imutils.imshow(unfiltered_reconstruction)
scipy.misc.imsave('unfilteredReconstruction.png', unfiltered_reconstruction)



"Use the FFT to translate the sinogram to the Frequency Domain and print the output"
print("Frequency Domain representation of sinogram")
frequency_domain_sinogram = fft_translate(sinogram)
imutils.imshow(frequency_domain_sinogram)
scipy.misc.imsave('frequencyDomainRepresentationOfSinogram.png', 
                  frequency_domain_sinogram)



"Filter the frequency domain projections by multiplying each one by the frequency domain ramp filter"
print("Frequency domain projections multipled with a ramp filter")
filtered_frequency_domain_sinogram = ramp_filter(frequency_domain_sinogram)
imutils.imshow(filtered_frequency_domain_sinogram)
scipy.misc.imsave('frequencyDomainProjectionsMultipledWithARampFilter.png', 
                  filtered_frequency_domain_sinogram)



"Use the inverse FFT to return to the spatial domain"
print("Spatial domain representation of ramp filtered sinogram")
filtered_spatial_domain_sinogram = inverse_fft_translate(filtered_frequency_domain_sinogram)
imutils.imshow(filtered_spatial_domain_sinogram)
scipy.misc.imsave('spatialDomainRepresentationOfRampFilteredSinogram.png', 
                  filtered_spatial_domain_sinogram)




"Re-construct the original 2D image by back-projecting the filtered projections"
print("Original, reconstructed image")
reconstructed_image = back_project(filtered_spatial_domain_sinogram)
imutils.imshow(reconstructed_image)
scipy.misc.imsave('originalReconstructedImage.png', 
                  reconstructed_image)


"Hamming-Windowed Ramp Filter"
print("Hamming-Windowed reconstructed image")
window = np.hamming(566)
hamming = reconstructed_image * window
imutils.imshow(hamming)
scipy.misc.imsave('hammingWindowedReconstructedImage.png', 
                  hamming)

