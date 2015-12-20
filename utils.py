from math import exp,log

def sigmoid(z):
	"""handel extreme condition by bouding sigmoid and 1- sigmoid larger than 1e -16 
	"""
	g = 1.0 / (1.0 + exp(-z))
	bound = exp(-16)
	g = min(1-bound, max(bound,g)) # 1-bound > g > bound
	
	return g

def logloss(act,pred):
	"""Bounded logloss
	"""
	bound = exp(-16) 
	pred = min(1-bound, max(bound,pred))
	return -(act * log(pred) + (1-act)*log(1-pred))	
