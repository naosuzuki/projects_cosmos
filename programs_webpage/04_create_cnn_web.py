import pandas as pd
import numpy
import os
import sys
import math

def create_web(mastercsv,jwstcsv,euclidcsv,hstcsv,outputhtml):
   df_master=pd.read_csv(mastercsv)
   df_jwst=pd.read_csv(jwstcsv,dtype={'jwstid':numpy.int64,'idstring':str})
   df_euclid=pd.read_csv(euclidcsv,dtype={'jwstid':numpy.int64,'idstring':str})
   df_hst=pd.read_csv(hstcsv,dtype={'jwstid':numpy.int64,'idstring':str})
   print(df_jwst)
   df_master["object_id"] = df_master["object_id"].astype(str).str.zfill(6)
   print(df_master)

# JWST as a base
   df1=df_jwst[df_jwst['telescope']=='jwst1']
   print(df_jwst)
# JWST + HST
   df_tmp=pd.merge(df1,df_hst,how='left',on='jwstid')
   print(df_tmp)
# JWST + HST + Euclid
   df_all=pd.merge(df_tmp,df_euclid,on='jwstid')
   print(df_all)
   df_all.to_csv('all.csv')

   df_cnn=pd.merge(df_master,df_all,right_on='idstring',left_on='object_id', how='left')
   print(df_cnn)

   sys.exit(1)
# JWST centered selection
   df=df_all.drop_duplicates(subset=['pngname_x'])
   print(df)
   ngroup=math.ceil(len(df)/1000.0)
   ngroup_string=str(ngroup)
   print('ngroup=',ngroup,ngroup_string.zfill(4))

   for n in range(ngroup):
      outputhtml='index'+str(n+1).zfill(4)+'.html'
      outputfile=open(outputhtml,'w')
      outputfile.write("<html>"+"\n")
      outputfile.write("<style>"+"\n")
      outputfile.write(".header {"+"\n")
      outputfile.write("  position: sticky;"+"\n")
      outputfile.write("  top: 0;"+"\n")
      outputfile.write("}"+"\n")
      outputfile.write("</style>"+"\n")
      outputfile.write(""+"\n")
      outputfile.write("<div class='header' id='myHeader'>"+"\n")
      outputfile.write("<img src='telescopetitle.png' align='left' width='100.0%'>"+"\n")
      outputfile.write("</div>"+"\n")
      outputfile.write(""+"\n")

#     outputfile.write("<img src='telescopetitle.png' align='left' width='96.0%'>"+"\n")
      istart=n*1000
      iend=min((n+1)*1000,len(df))
      print('istart=',istart,'iend=',iend)
      #sys.exit(1)
      #for i in range(len(df)):
      for i in range(istart,iend):
#      outputfile.write("<table width=\"100%\">"+"\n")
       jwstpng1=df.iloc[i]['pngname_x']
       jwstpng2=jwstpng1.replace('jwst1','jwst2')
       hstpng=df.iloc[i]['pngname_y']
       euclidpng=df.iloc[i]['pngname']
       outputfile.write("<center>"+"\n")
       outputfile.write("<img src="+"'"+"./jwst/"+jwstpng1+"'"+" align="+"'left' width="+"'"+"24.0%"+"'>"+"\n")  
       outputfile.write("<img src="+"'"+"./jwst/"+jwstpng2+"'"+" align="+"'left' width="+"'"+"24.0%"+"'>"+"\n")  
       outputfile.write("<img src="+"'"+"./hst/"+hstpng+"'"+" align="+"'left' width="+"'"+"24.0%"+"'>"+"\n")  
       outputfile.write("<img src="+"'"+"./euclid/"+euclidpng+"'"+" align="+"'left' width="+"'"+"24.0%"+"'>"+"\n")  
       outputfile.write("</center>")
       outputfile.write("<br>")
       outputfile.write("<hr width=\"100%\">")
      outputfile.write("</html>"+"\n")

jwstcsv='jwstpng.csv'
euclidcsv='euclidpng.csv'
hstcsv='hstpng.csv'
mastercsv='../csvfiles/supernova_cnn_candidates.csv'
outputhtml='cnn.html'
create_web(mastercsv,jwstcsv,euclidcsv,hstcsv,outputhtml)
