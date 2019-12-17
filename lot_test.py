######################################## 
## Define a grammar
######################################## 

from LOTlib.Grammar import Grammar
grammar = Grammar(start='EXPR')

grammar.add_rule('EXPR', '(%s + %s)', ['EXPR', 'EXPR'], 1.0)
grammar.add_rule('EXPR', '(%s * %s)', ['EXPR', 'EXPR'], 1.0)
grammar.add_rule('EXPR', '(float(%s) / float(%s))', ['EXPR', 'EXPR'], 1.0)
grammar.add_rule('EXPR', '(-%s)', ['EXPR'], 1.0)

# Now define how the grammar uses x. The string 'x' must
# be the same as used in the args below
grammar.add_rule('EXPR', 'x', None, 1.0) 

for n in xrange(1,10):
    grammar.add_rule('EXPR', str(n), None, 10.0/n**2)

from math import log
from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis

######################################## 
## Define the hypothesis
######################################## 

# define a 
class MyHypothesisX(LOTHypothesis):
    def __init__(self, **kwargs):
        LOTHypothesis.__init__(self, grammar=grammar, display="lambda x: %s", **kwargs)
    
    def __call__(self, *args):
        try:
            # try to do it from the superclass
            return LOTHypothesis.__call__(self, *args)
        except ZeroDivisionError:
            # and if we get an error, return nan
            return float("nan")

    def compute_single_likelihood(self, datum):
        if self(*datum.input) == datum.output:
            return log((1.0-datum.alpha)/100. + datum.alpha)
        else:
            return log((1.0-datum.alpha)/100.)

######################################## 
## Define the data
######################################## 

from LOTlib.DataAndObjects import FunctionData

# Now our data takes input x=3 and maps it to 12
# What could the function be?
data = [ FunctionData(input=[3], output=12, alpha=0.95) ]

######################################## 
## Actually run
######################################## 
from LOTlib.Inference.Samplers.MetropolisHastings import MHSampler
from LOTlib.SampleStream import *
## First let's make a bunch of hypotheses
from LOTlib.TopN import TopN

tn = TopN(1000) 

h0 = MyHypothesisX()
for h in MHSampler(h0, data, steps=100000): # run more steps
    tn.add(h)

# store these in a list (tn.get_all is defaultly a generator)
hypotheses = list(tn.get_all())

# Compute the normalizing constant
from LOTlib.Miscellaneous import logsumexp
z = logsumexp([h.posterior_score for h in hypotheses])

## Now compute a matrix of how likely each input is to go
## to each output
M = 20 # an MxM matrix of values
import numpy

# The probability of generalizing
G = numpy.zeros((M,M))

# Now add in each hypothesis' predictive
for h in hypotheses:
    print h 
    # the (normalized) posterior probability of this hypothesis
    p = numpy.exp(h.posterior_score - z)
    
    for x in xrange(M):
        output = h(x)
        
        # only keep those that are in the right range
        if 0 <= output < M:
            G[x][output] += p

# And show the output
print numpy.array_str(G, precision=3)