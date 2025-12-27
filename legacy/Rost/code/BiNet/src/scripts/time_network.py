import BiNet
from BiNet import network as network
from time import clock
import cProfile, pstats, StringIO
import pylab
import re
pylab.seed(0)
Ns = [4000,1000]
ps = pylab.ones((2,2))*0.5
ps[0,0] = 0.2
Ts = [1,1]
update_ratios = [1,2]
net = network.BalancedNetwork(Ns,ps,Ts,update_ratios=update_ratios,n_updates = 1)
w_in = pylab.ones((5000,1))*5
w_in[:4000]  =10.
input = pylab.ones((1,6000*20))

net.set_input_weights(w_in)
net.initialise_state(0.1)

	



pr = cProfile.Profile()
pr.enable()
output = net.forward(input,return_spiketimes = True)
pr.disable()
s = StringIO.StringIO()
sortby = 'time'
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
print s.getvalue()
print output.mean()