'''
Click-Through Rate Prediction Competition Python Solution
https://www.kaggle.com/c/avazu-ctr-prediction
@author Hetian Chen Dec 02, 2014

Example:
python main.py -t train.gz -p test.gz -s submission.csv --h 100  
'''
from datetime import datetime
from csv import DictReader
from utils import logloss
from PCFTRL import PCFTRL
import argparse
import gzip

def grid_search(train,holdout):
	'''
	Grid search of best hyperparameter combinations
	'''

	alpha = [0.01,0.1,1,10,100,1000]
	beta = [0.1,1,10]
	lambda1 = [0.001,0.01,0.1,1,1,10,100,1000]
	lambda2 = [0.001,0.01,0.1,1,1,10,100,1000]
	Dim = 2 ** 24  

	results = []
	for a in alpha:
		for b in beta:
			for l1 in lambda1:
				for l2 in lambda2:
					start = datetime.now()
					pcftrl = PCFTRL(a, b, l1, l2, Dim)
					loss = 0.
					count = 0
					for t, ID, x, y in data(train, Dim):
						if  t % holdout == 0: 
							pred = pcftrl.predict(x)
							loss += logloss(y,pred)
							count += 1
						else:
							pcftrl.learn(x,y)

					results.append((a,b,l1,l2,loss/count,pcftrl))
					print('\
						alpha: %f ,\
						beta: %f ,\
						lambda1: %f ,\
						lambda2: %f ,\
						holdout logloss: %f,\
						elapsed time: %s\
						'% (a,b,l1,l2,loss/count, str(datetime.now() - start)))

	results.sort(key=lambda tup:tup[4], reverse=False)
	params = results[0]
	a = params[0]
	b = params[1]
	l1 = params[2]
	l2 = params[3]
	loss = params[4]
	pcftrl = params[5]

	print('\
	best hyperparameters:\
	alpha: %f ,\
	beta: %f ,\
	lambda1: %f ,\
	lambda2: %f ,\
	holdout logloss: %f,\
	elapsed time: %s\
	'% (a,b,l1,l2,loss, str(datetime.now() - start)))

	return pcftrl

def write_output(pcftrl,test,submission):

	with open(submission, 'w') as outfile:
	    outfile.write('id,click\n')
	    for t, ID, x, y in data(test, pcftrl.Dim):
	        click = pcftrl.predict(x)
	        outfile.write('%s,%s\n' % (ID, str(click)))

def data(path, D):
    ''' GENERATOR: Apply hash-trick to the original csv row
                   and for simplicity, we one-hot-encode everything

        INPUT:
            path: path to training or testing file
            D: the max index that we can hash to

        YIELDS:
            ID: id of the instance, mainly useless
            x: a list of hashed and one-hot-encoded 'indices'
               we only need the index since all values are either 0 or 1
            y: y = 1 if we have a click, else we have y = 0
    '''

    for t, row in enumerate(DictReader(gzip.open(path))):
        # process id
        ID = row['id']
        del row['id']

        # process clicks
        y = 0.
        if 'click' in row:
            if row['click'] == '1':
                y = 1.
            del row['click']

        # turn hour really into hour, it was originally YYMMDDHH
        row['hour'] = row['hour'][6:]

        # build x
        x = [0]  # 0 is the index of the bias term
        for key in sorted(row):  # sort is for preserving feature ordering
            value = row[key]

            # one-hot encode everything with hash trick
            index = abs(hash(key + '_' + value)) % D
            x.append(index)

        yield t, ID, x, y


def myargs():
    
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description = 
""" 
Train model using Per-Coordinate FTRL-Proximal algorithm.
Hyperparameter tuning by holding out every N instance for validation.
Generate prediction results in the format of submission requirements. 

\nUsage is via:
\n
\n\t\tpython main.py -t <training file name> -p <testing file name> -s <submission file name>--h <holdout>
\n
""")
    parser.add_argument('-t', type = str)
    parser.add_argument('-p', type = str)
    parser.add_argument('-s', type = str)
    parser.add_argument('--h', default = 100, type = float)
  
    args = parser.parse_args()
    for v in vars(args).keys():
        print("%s => %s\n" % (v, str(vars(args)[v])))
    return args

def run():
	args = myargs()
	train = args.t
	test = args.p
	submission = args.s
	holdout = args.h
	pcftrl = grid_search(train,holdout)
	write_output(pcftrl,test,submission)

if __name__ == "__main__":
	run()





