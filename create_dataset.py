import pandas as pd
import numpy as np
import csv
import matplotlib.finance as f

fin_data=pd.read_csv('cleaned.csv')
symbols = []
for i,r in fin_data.iterrows():
	string=r['Symbol']
	symbols.append(string)

#resultFile = open('symbols.csv', 'wb')
#wr = csv.writer(resultFile, dialect='excel')
#wr.writerow(symbols)

d1=(2000, 1, 1)
d2=(2001, 1, 1)
variation=[]
for s in symbols:
	try:
		sp=f.quotes_historical_yahoo(s, d1, d2, asobject=False, adjusted=True)
		#third entry in each tuple is close, second is open
		var = [i[2] - i[1] for i in sp]
		var.insert(0, s)
		variation.append(var)
	except:
		variation.append([s])


np.savetxt('variation.csv', variation, delimiter=",")
