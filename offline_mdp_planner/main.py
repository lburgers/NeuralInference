from Bishop import *
import numpy as np

def main():

	obs = LoadObserver("Map", 0, 0)
	obs.Prepare()
	policy = obs.Plr.Policies[1]
	import pdb; pdb.set_trace()

if __name__ == "__main__":
	main()