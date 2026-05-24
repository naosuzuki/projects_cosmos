import pandas as pd
import numpy
import os
import sys
import math

def create_landingpage(ngroup):
      outputhtml='index.html'
      outputfile=open(outputhtml,'w')
      outputfile.write("<html>"+"\n")
      outputfile.write("<style>"+"\n")
      outputfile.write(".header {"+"\n")
      outputfile.write("  position: sticky;"+"\n")
      outputfile.write("  top: 0;"+"\n")
      outputfile.write("}"+"\n")
      outputfile.write(".pagination {"+"\n")
      outputfile.write("display: inline-block;"+"\n")
      outputfile.write("}"+"\n")
      outputfile.write(".pagination a {"+"\n")
      outputfile.write("color: black;"+"\n")
      outputfile.write("float: left;"+"\n")
      outputfile.write("padding: 8px 16px;"+"\n")
      outputfile.write("text-decoration: none;"+"\n")
      outputfile.write("transition: background-color .3s;"+"\n")
      outputfile.write("}"+"\n")      
      outputfile.write(".pagination a.active {"+"\n")      
      outputfile.write("background-color: #4CAF50;"+"\n")
      outputfile.write("color: white;"+"\n")
      outputfile.write("}"+"\n")
      outputfile.write(".pagination a:hover:not(.active) {background-color: #ddd;}"+"\n")
      outputfile.write("</style>"+"\n")
      outputfile.write(""+"\n")

      outputfile.write("<body>"+"\n")
      outputfile.write("<div class='pagination'>"+"\n")

      outputfile.write("<a href='index0001.html'>&laquo;</a>"+"\n")

      for n in range(ngroup):
         outputhtml='index'+str(n+1).zfill(4)+'.html'
         outputfile.write("<a href='index"+str(n+1).zfill(4)+".html'>"+str(n+1)+"</a>"+"\n") 

      outputfile.write("<a href='index"+str(ngroup+1).zfill(4)+".html'>&raquo;</a>"+"\n")
      outputfile.write("</div>"+"\n")
      outputfile.write("</body>"+"\n")
      outputfile.write("</html>"+"\n")

def create_website(csvfile):
#   df_master=pd.read_csv(mastercsv)
#   df_jwst=pd.read_csv(jwstcsv,dtype={'jwstid':numpy.int64,'idstring':str})
#   df_euclid=pd.read_csv(euclidcsv,dtype={'jwstid':numpy.int64,'idstring':str})
#   df_hst=pd.read_csv(hstcsv,dtype={'jwstid':numpy.int64,'idstring':str})

# JWST centered selection
   df=pd.read_csv(csvfile,dtype={'jwstid':numpy.int64,'jwststr':str})
   print(df)
   ngroup=math.ceil(len(df)/1000.0)
   ngroup_string=str(ngroup)
   print('ngroup=',ngroup,ngroup_string.zfill(4))

# Landing Page
   create_landingpage(ngroup)

   for n in range(ngroup):
      outputhtml='index'+str(n+1).zfill(4)+'.html'
      outputfile=open(outputhtml,'w')
      outputfile.write("<html>"+"\n")
      outputfile.write("<style>"+"\n")
      outputfile.write(".header {"+"\n")
      outputfile.write("  position: sticky;"+"\n")
      outputfile.write("  top: 0;"+"\n")
      outputfile.write("}"+"\n")
      outputfile.write(".pagination {"+"\n")
      outputfile.write("display: inline-block;"+"\n")
      outputfile.write("}"+"\n")
      outputfile.write(".pagination a {"+"\n")
      outputfile.write("color: black;"+"\n")
      outputfile.write("float: left;"+"\n")
      outputfile.write("padding: 8px 16px;"+"\n")
      outputfile.write("text-decoration: none;"+"\n")
      outputfile.write("transition: background-color .3s;"+"\n")
      outputfile.write("}"+"\n")      
      outputfile.write(".pagination a.active {"+"\n")      
      outputfile.write("background-color: #4CAF50;"+"\n")
      outputfile.write("color: white;"+"\n")
      outputfile.write("}"+"\n")
      outputfile.write(".pagination a:hover:not(.active) {background-color: #ddd;}"+"\n")
      outputfile.write("</style>"+"\n")
      outputfile.write(""+"\n")

# Header
      outputfile.write("<div class='header' id='myHeader'>"+"\n")

      outputfile.write("<body>"+"\n")
      outputfile.write("<div class='pagination'>"+"\n")
      outputfile.write("<a href='index0001.html'>&laquo;</a>"+"\n")

      for m in range(ngroup):
         outputhtml='index'+str(m+1).zfill(4)+'.html'
         outputfile.write("<a href='index"+str(m+1).zfill(4)+".html'>"+str(m+1)+"</a>"+"\n") 

      outputfile.write("<a href='index"+str(ngroup+1).zfill(4)+".html'>&raquo;</a>"+"\n")
      outputfile.write("</div>"+"\n")
      outputfile.write("</body>"+"\n")

      outputfile.write("<img src='telescopenames4.png' align='left' width='100.0%'>"+"\n")
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
       jwstpng1=df.iloc[i]['jwst1png']
       jwstpng2=df.iloc[i]['jwst2png']
       hstpng=df.iloc[i]['hstpng']
       euclidpng=df.iloc[i]['euclidpng']
       ra=df.iloc[i]['ra']
       dec=df.iloc[i]['dec']
       jwstid=df.iloc[i]['jwstid']
       z=df.iloc[i]['z']
       outputfile.write("<left>"+"\n")
       outputfile.write("<tbody><tr><h2>  ID="+"%-7i"%(jwstid)+\
                        "    RA "+ra+" DEC "+dec+"  z="+"%5.3f"%(z)+"</tr></h2>"+"\n")
       outputfile.write("</left>"+"\n")
       outputfile.write("<center>"+"\n")
       outputfile.write("<img src="+"'"+"./jwst/"+jwstpng1+"'"+" align="+"'left' width="+"'"+"24.0%"+"'>"+"\n")  
       outputfile.write("<img src="+"'"+"./jwst/"+jwstpng2+"'"+" align="+"'left' width="+"'"+"24.0%"+"'>"+"\n")  
       outputfile.write("<img src="+"'"+"./hst/"+hstpng+"'"+" align="+"'left' width="+"'"+"24.0%"+"'>"+"\n")  
       outputfile.write("<img src="+"'"+"./euclid/"+euclidpng+"'"+" align="+"'left' width="+"'"+"24.0%"+"'>"+"\n")  
       outputfile.write("</center>")
       outputfile.write("<br>")
       outputfile.write("<hr width=\"100%\">")
      outputfile.write("</html>"+"\n")

#jwstcsv='jwstpng.csv'
#euclidcsv='euclidpng.csv'
#hstcsv='hstpng.csv'
mastercsv='../csvfiles/COSMOSWeb_masterv3.1_mag25cut.csv'
csvfile='../csvfiles/jwsthsteuclid.csv'
#create_web(mastercsv,jwstcsv,euclidcsv,hstcsv,outputhtml)
create_website(csvfile)

