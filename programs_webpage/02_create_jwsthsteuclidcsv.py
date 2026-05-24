import pandas as pd
import numpy
import os
import sys
import math
from astropy import units as u
from astropy.coordinates import SkyCoord

def create_web(mastercsv,jwstcsv,euclidcsv,hstcsv,outputcsv):
   df_master=pd.read_csv(mastercsv)
#   print(df_master)
#   sys.exit(1)
   df_jwst=pd.read_csv(jwstcsv,dtype={'jwstid':numpy.int64,'idstring':str})
   df_euclid=pd.read_csv(euclidcsv,dtype={'jwstid':numpy.int64,'idstring':str})
   df_hst=pd.read_csv(hstcsv,dtype={'jwstid':numpy.int64,'idstring':str})

# JWST as a base
#   print(df_jwst)
   df1=df_jwst[df_jwst['telescope']=='jwst1']
   df2=df1[['idstring']]
   idlist=df2['idstring'].unique().tolist()

   jwst1list=[]  ; jwst2list=[] ; hstlist=[] ; euclidlist=[]
   jwstidlist=[] ; jwstidstringlist=[]  
   ralist=[] ; declist=[] ; zlist=[] ; tilelist=[]
# Loop i per object ID
   for i in range(len(idlist)):
# Counting the Number of Tiles
      jwstnum=len(df1[df1['idstring']==idlist[i]])
# Euclid
      euclidnum=len(df_euclid[df_euclid['idstring']==idlist[i]]) 
      if(euclidnum==0): 
         continue
      else:
         euclidname=df_euclid[df_euclid['idstring']==idlist[i]]['pngname'].iloc[0]
# HST Tiles
      hstnum=len(df_hst[df_hst['idstring']==idlist[i]])
      if(hstnum==0): continue
#     print(i,idlist[i],jwstnum,euclidnum,hstnum)
        
# Loop j per tile
      for j in range(jwstnum):
         jwstname=df_jwst[(df_jwst['idstring']==idlist[i]) & (df_jwst['telescope']=='jwst1')]['pngname'].iloc[j]
         jwstid=df_jwst[(df_jwst['idstring']==idlist[i]) & (df_jwst['telescope']=='jwst1')]['jwstid'].iloc[j]
         if(jwstnum<=hstnum):
           hstname=df_hst[df_hst['idstring']==idlist[i]]['pngname'].iloc[j] 
         elif(j>hstnum):
           hstname=df_hst[df_hst['idstring']==idlist[i]]['pngname'].iloc[0] 
# RA DEC
         radeg=df_master[df_master['id']==jwstid]['ra'].iloc[0]
         decdeg=df_master[df_master['id']==jwstid]['dec'].iloc[0]
         radec=SkyCoord(ra=radeg*u.degree, dec=decdeg*u.degree)
         radecstring=radec.to_string('hmsdms')
         radecstring=radecstring.replace('h',':')
         radecstring=radecstring.replace('m',':')
         radecstring=radecstring.replace('d',':')
         radecstring=radecstring.replace('s','')
         rastr=radecstring[0:12]
         dectmp=radecstring.split(' ')[1]
         decstr=dectmp[0:12]
         ralist.append(rastr) ; declist.append(decstr)
# Tile
         tilestr=jwstname.split('_')[1]
         tilelist.append(tilestr)
# Redshift
         redshift=df_master[df_master['id']==jwstid]['LP_zfinal'].iloc[0]
         if(redshift<0): redshift=0.0
         zlist.append(redshift)
# IDs
         jwstidlist.append(jwstid)
         jwstidstringlist.append(idlist[i])

         jwst1list.append(jwstname)
         jwst2list.append(jwstname.replace('jwst1','jwst2'))
         hstlist.append(hstname)
         euclidlist.append(euclidname)
         print(jwstid,idlist[i],tilestr,rastr,decstr,redshift,jwstname,hstname,euclidname)

#   column_values=['jwst1png','jwst2png','hstpng','euclidpng']
   column_values=['jwstid','jwstidstr','tile','ra','dec','z','jwst1png','jwst2png','hstpng','euclidpng']
   arr=numpy.array([jwstidlist,jwstidstringlist,tilelist,ralist,declist,zlist,\
                   jwst1list,jwst2list,hstlist,euclidlist])
   arr=arr.transpose()
   print(arr)
   df=pd.DataFrame(data=arr,columns=column_values)
   print(df)
   df.to_csv(outputcsv,index=False)

jwstcsv='../csvfiles/jwstpng.csv'
euclidcsv='../csvfiles/euclidpng.csv'
hstcsv='../csvfiles/hstpng.csv'
mastercsv='../csvfiles/COSMOSWeb_masterv3.1_mag25cut.csv'
outputcsv='../csvfiles/jwsthsteuclid.csv'
create_web(mastercsv,jwstcsv,euclidcsv,hstcsv,outputcsv)
