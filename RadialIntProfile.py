#/usr/bin/env python
#
"""
RadialIntProfile.py: script to calculate the radial intensity profile
of a disk image averaging in rings (similar to irings in AIPS).

Developed by Enrique Macias and Carlos Carrasco-Gonzalez.

# ========================================================================================
"""
import numpy as np
import scipy.optimize
import astropy.io.fits as pyfits
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def rad_profile(im_name,theta_i,theta_pa,rmin,rmax,dr,cent=(1,1),\
                ring_width=0.0,phi_min=0.0,phi_max=2.*np.pi,im_rms=-1,\
                doNorm=False,expNorm=1.5,doModel=False,outfile='',\
                doPlot=True,color='blue',dist=''):
    """
    Input parameters:
    - imname: name of the image, in fits format.
    - theta_i, theta_pa: inclination and position angle of the source, in degrees.
    - rmin, rmax: minimum and maximum radii for the radial profile, in arcsec.
    - dr: separation between rings, in arcsec.
    - cent: tuple with the position of the center of the source (in pixels),
    i.e. where the annular eliptical rings will be centered.
    Optional:
    - ring_width: width of the rings, in arcsec. If not given as input, it will
    be set to one third of the beam size.
    - phi_min, phi_max (defaults are 0 and 2pi): Minimum and maximum azimuthal
    angles, in case particular angles want to be selected when averaging the emission.
    - im_rms (default is -1): rms of the map. If not provided, it will be calculated
    from the residual of the image after subtracting the averaged emission in each ring.
    - doNorm (default is False), expNorm (default is 1.5): if doNorm = True,
    the profile will be normalized, also dividing the intensity by r**expNorm.
    - doModel (default is False): If True, a "model" image will be created with
    the emission of each ring, together with an image of the uncertainty in each ring.
    - outfile (default is the image name, without extension): Name of the output
    files (.txt and .pdf).
    - doPlot (default is True): If True, a plot will be created with the profile.
    - color (default is blue): Color of the plot.
    - dist: distance to the source, in pc. It will only be necessary if a twin x axis
    with the distances in au is wanted.
    """
# We open the image
    imObj = pyfits.open(im_name)
    Header = imObj[0].header
    Image = imObj[0].data[0,0,:,:]
    imObj.close()

# We retreat some information from the header
    try:
        if Header['cunit1'] == 'deg':
            csize = abs(Header['cdelt1'])*3600. # Cellsize (")
        elif Header['cunit1'] == 'rad':
            csize = abs(Header['cdelt1'])*180./np.pi*3600. # Cellsize (")
    except:
        # If cunit1 is not provided, we assume that cdelt is given in deg
        print("cunit not provided in the header. Assuming cdelt is given in degrees.")
        csize = abs(Header['cdelt1'])*3600. # Cellsize (")
    IntUnit = Header['bunit'] # Brightness pixel units of image

    bmaj = Header['bmaj']*3600. # Beam major axis (")
    bmin = Header['bmin']*3600. # Beam minor axis (")

    nx = Header['naxis1'] # Number of pixels in the x axis
    ny = Header['naxis2'] # Number of pixels in the y axis

# We build arrays for the x and y positions in the image
    xarray = Image*0.
    yarray = Image*0.

    for i in range(nx):
        for j in range(ny):
            xarray[j,i] = i - cent[0]
            yarray[j,i] = j - cent[1]

# We rotate these arrays according to the PA of the source
# The input angle should be a PA (from N to E), but we subtract 90 degrees for this rotation
    theta_pa = theta_pa - 90.
    xrot = xarray * np.cos(-theta_pa*np.pi/180.) - yarray * np.sin(-theta_pa*np.pi/180.)
    yrot = xarray * np.sin(-theta_pa*np.pi/180.) + yarray * np.cos(-theta_pa*np.pi/180.)

# We rotate the y positions according to the inclination of the disk
    yrot = yrot / np.cos(theta_i*np.pi/180.)

# Array with the radii
    rrot = np.sqrt(xrot**2 + yrot**2) * csize

# Array with the azimuthal angles
# (yarray and xarray ar used so that phi is the azimuthal angle in the plane of the sky)
    phi=np.arctan2(yarray,xarray)
# We make all the angles positive
    phi[np.where(phi<0.0)] += 2.*np.pi

# Vector of radii for the position of the annular rings, from rmin to rmax
    nr = (rmax - rmin) / dr # Number of radii
    radii = (np.array(range(int(nr) + 1)) / nr) * (rmax - rmin) + rmin + dr/2.

# For each ring, we will calculate its average intensity and the uncertainty
    IntAv=[]
    IntErr=[]
    # In case we want to create an image with the "model"
    if doModel or (im_rms < 0):
        Model = xarray
        Model[:,:] = np.nan
        ErrModel = Model
    # We start averaging the emission
    if ring_width == 0.0:
        ring_width = np.sqrt(bmaj * bmin) / 3. # a third of the approximate beam size
    for r in radii:
        r0 = r - ring_width / 2.
        r1 = r + ring_width / 2.
        if r0 < 0.0:
            r0 = 0.0

        Ring = Image[(rrot>=r0) & (rrot<r1) & (phi>=phi_min) & (phi<=phi_max)]
        kAver = np.average(Ring) # Average within the ring

        Abeam = np.pi * bmaj * bmin/(4.0 * np.log(2.0)) # Area of the beam
        Aring = np.pi * np.cos(theta_i * np.pi/180.) * (r1**2 - r0**2) # Area of ring
        nbeams=Aring/Abeam # Number of beams in the ring
        if nbeams < 1.0:
            nbeams=1.0 # The error can never be lower than one rms
        kErr=1.0/np.sqrt(nbeams)
        # In case we want to create an image with the "model"
        if doModel or (im_rms < 0):
            Model[(rrot>=r0) & (rrot<r1)] = kAver
            ErrModel[(rrot>=r0) & (rrot<r1)] = kErr

        IntAv.append(kAver) # We save the averaged intensity in the ring
        IntErr.append(kErr) # We save the uncertainty in the ring

    radii = np.array(radii)
    IntAv = np.array(IntAv)
    IntErr = np.array(IntErr)

    # If we didn't provide an rms, it will calculate it from the residuals
    if im_rms < 0:
        Resid = Image - Model
        im_rms = np.sqrt(np.average(Resid[(rrot>rmin) & (rrot<rmax)]**2.0))
    # We multiply the error by the rms, and write the model images if set so
    IntErr = IntErr * im_rms
    if doModel:
        ErrModel = ErrModel * im_rms
        IntProfImObj = pyfits.writeto(imname[:-5] + '.Model.fits',Model,Header,clobber=True)
        IntProfImObj = pyfits.writeto(imname[:-5] + '.ModelErr.fits',ErrModel,Header,clobber=True)

    # In case we want to normalize the profile:
    if doNorm:
        IntAv = IntAv / np.power(radii,expNorm)
        IntErr = IntErr / np.power(radii,expNorm)
        maxint = np.max(IntAv)
        IntAv = IntAv / maxint
        IntErr = IntErr / maxint

# We write the output result in a file
    if outfile == '':
        outfile = im_name[:-5]
    f = open(outfile + '.txt',"w")
    print(' Radius    Av.Int.     Error       S/N    ')
    print('  (")          ('+IntUnit+')')
    f.write(' Radius    Av.Int.     Error       S/N    \n')
    f.write('  (")          ('+IntUnit+') \n')
    for r,A,ErrA in zip(radii,IntAv,IntErr):
        f.write("%f %g %g %f \n" %(r,A,ErrA,A/ErrA))
        print(r,A,ErrA,A/ErrA)
    f.close()

# We make a plot with the profile
    if doPlot:
        if doNorm:
            ytitle = 'Average Normalized Intensity'
        else:
            # If the units of the image are Jy/beam, we will make the plot in mJy/beam
            if IntUnit == 'Jy/beam':
                ytitle = 'Average Intensity (mJy/beam)'
                IntAv = IntAv * 1000.0 #mJy/beam
                IntErr = IntErr * 1000.0 #mJy/beam
            else:
                ytitle = 'Average Intensity ('+IntUnit+')'

        fig = plt.figure()
        ax1 = fig.add_subplot(1,1,1)
        ax1.set_ylabel(ytitle,fontsize=15)
        ax1.set_xlabel('Radius (arcsec)',fontsize=15)
        ax1.set_xlim([0.0,rmax])

        ax1.fill_between(radii,IntAv+IntErr,IntAv-IntErr,facecolor=color)
        ax1.plot(radii,IntAv,'k--')
        ax1.set_ylim(bottom=-0.002)

        if dist != '':
            twax1 = ax1.twiny()
            twax1.set_xlim([0.0,rmax*dist])
            twax1.xaxis.set_minor_locator(ticker.MultipleLocator(5))
            twax1.xaxis.set_major_locator(ticker.MultipleLocator(20))
            twax1.set_xlabel('Radius (au)',fontsize=15)

        plt.savefig(outfile + '.pdf',dpi = 650)
        plt.close(fig)

    # Returns arrays with radii, integrated intensity and uncertainty
    return radii, IntAv, IntErr
