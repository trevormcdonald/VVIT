import numpy as np
import pandas as pd

fin_data=pd.ExcelFile('data/bigdata (1).xlsx')
sheet1 = fin_data.parse("Sheet1")
price = fin_data.parse("Price")
pe = fin_data.parse("PE")
ps = fin_data.parse("PS")
cap = fin_data.parse("Cap")
pb = fin_data.parse("PB")
dividend = fin_data.parse("Dividend")

sheets=[price, pe, ps, cap, pb, dividend]
names=["Price", "PE", "PS", "Cap", "PB", "Dividend"]
mask= True
for s in sheets:
	mask = mask & s.notnull()
#mask = price.notnull() & pe.notnull() & ps.notnull() & cap.notnull() & pb.notnull() & dividend.notnull()

for i, s in enumerate(sheets):
	df = s.where(mask, other=0)
	s.to_csv('data/cleaned_' + names[i] + '.csv')
	

