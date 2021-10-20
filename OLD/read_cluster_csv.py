import csv
import numpy as np


def module_sizes(mydict):
	length = []
	for val in mydict.values(): 
		temp = 0 
		for i in range(56): 
			if i<10: 
				if ' '+str(i) in val: 
					temp +=1 
			else: 
				if str(i) in val: 
					temp += 1 
		length.append(temp) 
	return length

def gene_persistence(mydict):
	count = np.zeros(56)
	for val in mydict.values(): 
		for i in range(56): 
			if i<10: 
				if ' '+str(i) in val: 
					count[i]+= 1 
			else: 
				if str(i) in val: 
					count[i]+= 1
	return count


	


filename = 'hk_cluster_clst.csv'

with open(filename, mode='r') as infile: 
	reader = csv.reader(infile) 
	mydict = {rows[0]:rows[1] for rows in reader} 