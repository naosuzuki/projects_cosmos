import pandas as pd
import glob
import os
import sys
import numpy

jwstpngdir='/Users/suzuki/data/JWST/CosmosWeb/v0.8_png/'
euclidpngdir='/Users/suzuki/data/Euclid/COSMOS_OTF/NISP_png/'
hstpngdir='/Users/suzuki/data/HST/cosmosacs/original_png/'

jwstpngdir='/Users/suzuki/data/JWST/web/jwst/'
euclidpngdir='/Users/suzuki/data/JWST/web/euclid/'
hstpngdir='/Users/suzuki/data/JWST/web/hst/'

def create_pnglistcsv(outputcsv,flag):
   cwd=os.getcwd()
   if(flag=='jwst'): os.chdir(jwstpngdir)
   if(flag=='euclid'): os.chdir(euclidpngdir)
   if(flag=='hst'): os.chdir(hstpngdir)

   pnglist=sorted(glob.glob('*png'))
   print(pnglist)
   os.chdir(cwd)

   arr_name=["" for i in range(len(pnglist))]
   arr_id=["" for i in range(len(pnglist))]
   arr_telescope=["" for i in range(len(pnglist))]
   arr_pngname=["" for i in range(len(pnglist))]
   if(flag=='jwst' or flag=='hst'):
      arr_tile=["" for i in range(len(pnglist))]
      column_values=['jwstid','idstring','tile','telescope','pngname']
   if(flag=='euclid'):
      arr_tile=["" for i in range(len(pnglist))]
      column_values=['jwstid','idstring','telescope','pngname']

   for i in range(len(pnglist)):
     print(pnglist[i])
     idstring=pnglist[i].split('_')[0] 
     idnumber=int(idstring)

     if(flag=='jwst' or flag=='hst'):
        tmp=pnglist[i].split('_')[2]
        telescope=tmp.split('.')[0]
        tile=pnglist[i].split('_')[1]
        arr_tile[i]=tile
     if(flag=='euclid'):
        print(pnglist[i])
        tmp=pnglist[i].split('_')[1]
        telescope=tmp.split('.')[0]
#    print(idnumber,idstring,tile)

     arr_id[i]=idnumber
     arr_name[i]=idstring
     arr_telescope[i]=telescope
     arr_pngname[i]=pnglist[i]

   if(flag=='jwst' or flag=='hst'):
      arr=numpy.array([arr_id,arr_name,arr_tile,arr_telescope,arr_pngname])
   if(flag=='euclid'):
      arr=numpy.array([arr_id,arr_name,arr_telescope,arr_pngname])
   arr=arr.transpose()
   df=pd.DataFrame(data=arr,columns=column_values)
   print(arr)
   print(df)
   df.to_csv(outputcsv,index=False)

flag='hst'
outputcsv='hstpng.csv'
create_pnglistcsv(outputcsv,flag)

flag='euclid'
outputcsv='euclidpng.csv'
create_pnglistcsv(outputcsv,flag)

flag='jwst'
outputcsv='jwstpng.csv'
create_pnglistcsv(outputcsv,flag)
