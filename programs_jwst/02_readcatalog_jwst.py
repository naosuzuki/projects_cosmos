import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
import sys
#from gammapy.maps import Map
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.nddata.utils import Cutout2D
import numpy
import matplotlib.pyplot as plt
import matplotlib
from astropy.visualization import make_rgb, ManualInterval
from astropy.io import fits
#from astropy.utils.data import get_pkg_data_filename
#from astropy.visualization import (MinMaxInterval, SqrtStretch, ImageNormalize, LogStretch)
#from astropy.visualization import LuptonAsinhStretch
#import glob

jwstpng='/Users/suzuki/data/Euclid/COSMOS_OTF/NISP_png/'
jwstdir='/Users/suzuki/data/Euclid/COSMOS_OTF/NISP/'
jwstdir='/Users/suzuki/data/HST/clutch/v0.2/'
jwstpng='/Users/suzuki/data/HST/clutch/v0.2_png/'

jwstdir='/Users/suzuki/data/JWST/CosmosWeb/v0.8/'
jwstpng='/Users/suzuki/data/JWST/CosmosWeb/v0.8_png/'

def read_catalog(cosmoswebcsv,jwstfitscsv):
   df=pd.read_csv(cosmoswebcsv)
   dffits=pd.read_csv(jwstfitscsv)
   print(df)

#   for i in range(len(df)):
   for i in range(10):
#   for i in range(1):
#   for i in range(10000):
     print('Working on ',i)
     cosmosid=str(df.loc[i,'id'])
#    print(pngname)
     ra=df.loc[i,'ra']
     dec=df.loc[i,'dec']
     z=df.loc[i,'LP_zfinal']
     magf150=df.loc[i,'MAG_MODEL_F150W']
     magf277=df.loc[i,'MAG_MODEL_F277W']
     mass=df.loc[i,'LP_mass_minchi2']
     age=df.loc[i,'LP_age_minchi2']

     for j in range(len(dffits)):
        fitsname_1=dffits.loc[j,'fits1']
        [flag]=check_fits(fitsname_1,ra,dec)
        if(flag):
          [tilestring]=find_tilenumber(fitsname_1)
          pngname_1=jwstpng+cosmosid.zfill(6)+'_'+tilestring+'_jwst1.png'
          pngname_2=jwstpng+cosmosid.zfill(6)+'_'+tilestring+'_jwst2.png'
          objtitle=cosmosid.zfill(6)+' '+tilestring

          fitsname_2=dffits.loc[j,'fits2']
          fitsname_3=dffits.loc[j,'fits3']
          fitsname_4=dffits.loc[j,'fits4']
          #postion = SkyCoord()
          position = SkyCoord(ra,dec,unit="deg",frame='icrs')
#          print(position)
          count=0
          for fitsfilename in [fitsname_1, fitsname_2, fitsname_3, fitsname_4]:
             count+=1
             hdul=fits.open(jwstdir+fitsfilename)
             hdr=hdul[0].header
             wcs=WCS(hdr)
             # 6 arcsec cutout
             cutout = Cutout2D(hdul[0].section,position,size=6.0*u.arcsec,wcs=wcs)
             hdul.close()
# Check Image Size
             nx=cutout.data.shape[0]
             ny=cutout.data.shape[1]
#             print('count=',count,'fitsfilename',fitsfilename)
#             print(cutout.data)
#             print(cutout.data.shape)
#             print('x,y',count,nx,ny)
#             sys.exit(1)
             if((nx < 200) or (ny < 200)): continue
             #ny=cutout.data.shape[1]
             if(count==1): cutout_1=cutout.data ; del cutout 
             if(count==2): cutout_2=cutout.data ; del cutout
             if(count==3):  
               cutout_3=cutout.data ; del cutout ; magflag=0 ; mag=magf150
               create_colorimg_jwst(cutout_1,cutout_2,cutout_3,pngname_1,objtitle,z,mag,magflag,mass,age)
               del magflag
             if(count==4):  
               cutout_4=cutout.data ; del cutout ; magflag=1 ; mag=magf277
               create_colorimg_jwst(cutout_2,cutout_3,cutout_4,pngname_2,objtitle,z,mag,magflag,mass,age)
               del magflag

#     print(i,ra,dec,z)

def find_tilenumber(fitsfilename):
    tilestring=fitsfilename.split('_')[5]
    return [tilestring]

def check_fits(fitsfilename,ra,dec):
# Checking the coordinates are in FITS fiel
   f=fits.open(jwstdir+fitsfilename)
   hdr=f[0].header   
   f.close()
   naxis1=hdr['NAXIS1']
   naxis2=hdr['NAXIS2']
   w=WCS(f[0].header)
   x,y=w.all_world2pix(ra,dec,1)
#  Check if RA & DEC are in the FITS
   if(x>0 and x<naxis1 and y>0 and y<naxis2): 
      flag=True
   else:
      flag=False
#   print(ra,dec,x,y,flag)
   return [flag]

#def create_colorimg(fits1,fits2,fits3,pngname,objtitle):
def create_colorimg_jwst(data_1,data_2,data_3,pngname,objtitle,z,mag,magflag,mass,age):
# Read in the three images downloaded from here:
#   b = fits.getdata(fits1)
#   g = fits.getdata(fits2)
#   r = fits.getdata(fits3)
   b = data_1
   g = data_2
   r = data_3

# Use the maximum value of the 99.5% percentile over all three filters
# as the maximum value:
#   pctl = 99.5
   minimum = 0.0001
#  minimum = 0.01

# Replace negative value
   b=numpy.where(b>minimum,b,minimum)
   g=numpy.where(g>minimum,g,minimum)
   r=numpy.where(r>minimum,r,minimum)

   b=numpy.sqrt(b)
   g=numpy.sqrt(g)
   r=numpy.sqrt(r)

   for img in [b,g,r]:
      maximum = 0.001
# This is for JWST 30mas image
#     val=np.amax(img[80:89,80:89])
# This is for Euclid
#     val=numpy.amax(img[25:35,25:35])
# This is for HST/JWST 30mas image 200pix x 200pix
      val=numpy.amax(img[95:105,95:105])
#      val = np.percentile(img,pctl)
      if val > maximum: maximum = val
#      print('measured max=',maximum)
   #  if maximum < 0.1: maximum =1.0
# Needs Adjustment For JWST
   if(maximum<1.0): 
      maximum=1.0
   else:
      maximum*=0.85
#   print(objtitle,'min=',minimum,'max=',maximum)

   rgb = make_rgb(r, g, b, interval=ManualInterval(vmin=minimum, vmax=maximum))

   matplotlib.use('Agg')
# Font to be Times
   plt.rc('font',family='serif')
   plt.xlim(0,199)
   plt.ylim(0,199)
#  plt.subplots_adjust(top=1,bottom=0,right=1,left=0,hspace=0,wspace=0)
#   ax = plt.gca()
#   ax.set_aspect('equal', adjustable='box')
#   fig,ax = plt.subplots(subplot_kw={'aspect': 'equal'})
#   ax.axis('off')
#   plt.axis('off')
#   plt.aspect('equal','box')
#   plt.margins(x=0)
#   plt.margins(y=0)
#   plt.rcParams["font.family"]="Times New Roman"
   plt.axis('off')
# For Euclid
#   xmin=1.0 ; dx=60. ; ymin=1.0 ; dy=60.
# For HST/JWST
   xmin=1.0 ; dx=200. ; ymin=1.0 ; dy=200.
   plt.text(xmin+dx*0.07,ymin+0.10*dy,"%-9s"%(objtitle),color='white',fontsize=18)
   plt.text(xmin+dx*0.40,ymin+0.10*dy,"z="+"%5.3f"%(z),color='white',fontsize=18)
   if(magflag==0): 
     plt.text(xmin+dx*0.67,ymin+0.10*dy,r"m$_{150}$="+"%5.2f"%(mag),color='white',fontsize=16)
   elif(magflag==1): 
     plt.text(xmin+dx*0.67,ymin+0.10*dy,r"m$_{277}$="+"%5.2f"%(mag),color='white',fontsize=16)

   plt.text(xmin+dx*0.67,ymin+0.02*dy,"M="+"%5.2f"%(mass)+r" M$_{\odot}$",color='white',fontsize=16)
#   plt.text(xmin+dx*0.72,ymin+0.02*dy,r"t$_{age}$="+"%5.2f"%(age),color='white',fontsize=18)

   plt.imshow(rgb, origin='lower')
   plt.tight_layout()
   plt.savefig(pngname,bbox_inches='tight',pad_inches=0)
   plt.clf()
   plt.close()

#hstclutcfits3csv='../csv/EUCLIDfits.csv'
jwstfitscsv='../csv/jwstfits.csv'
cosmoswebcsv='../csv/COSMOSWeb_masterv3.1_mag25cut.csv'
read_catalog(cosmoswebcsv,jwstfitscsv)
