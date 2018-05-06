import os
import csv
import numpy as np

#to_combine_file = ['prob-ck5000-lr00001-1524450365.csv', 'prob-ck5500-lr00001-1524497893.csv', 'prob-ck7000-lr00001-1524497893.csv', 
#	'prob-ck9000-lr00001-1524497893.csv', 'prob-ck11000-lr00001-1524497893.csv', 'prob-ck11500-lr001.csv', 'prob-ck10500-lr0001-L20.4.csv', 
#	'prob-ck5500-lr00001-1524450365.csv', 'prob-ck11000-lr0001-1524290099.csv', 'prob-ck8500-lr00001-1524368649.csv', 'prob-ck11500-lr00001-1524368649.csv',
#	'prob-ck17500-lr00001-1524368649.csv']

to_combine_file = [_ for _ in os.listdir('.') if _.endswith('.csv')] # and _[-8:-4] == '2748'
#print(to_combine_file)
#input()
output_dir = 'output'

def from_file2dict(csvfile):
	'''
	transform csv file to dict file
	'''
	to_dict = {}
	with open(csvfile, 'r') as f:
		reader = csv.DictReader(f)
		for mini_dict in reader:
			item_id = mini_dict['id']
			item_pred = float(mini_dict['pred'])
			to_dict[item_id] = item_pred

	return to_dict

def main():
	# randomly generate a coff combination
	random_coff = np.random.uniform(low=0., high=1., size=(len(to_combine_file),))
	random_coff = random_coff / random_coff.sum()

	# generate all file dict
	file_dicts = []
	for file in to_combine_file:
		file_dicts.append(from_file2dict(file))

	# linear combine all the predictions
	final_dict = {}
	for ind in file_dicts[-1].keys():
		avg = 0.
		for i in range(len(to_combine_file)):
			avg += file_dicts[i][ind]
		avg /= len(to_combine_file)

		final_dict[ind] = avg

	# write averaged file to format we want
	with open(os.path.join(output_dir, 'final.csv'), 'w') as f:
		writer = csv.writer(f)
		writer.writerows([['id', 'pred']])
		for ind in final_dict.keys():
			writer.writerows([[ind, final_dict[ind]]])


if __name__ == '__main__':
	main()
