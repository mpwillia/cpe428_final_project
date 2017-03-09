
#try:
#    import cPickle as pickle
#except:
#    import pickle
import dill as pickle

import os

def save(net, name = None):
    if name is None:
        name = net.network_name
    
    vars_name = name + ".ckpt"
    net_name = name + ".pkl"

    vars_path = os.path.join("./", vars_name)
    net_path = os.path.join("./", net_name)
    
    print("Saving Network")
    net.save_variables(vars_path)

    with open(net_path, 'wb') as f:
        pickle.dump(net, f)


def load(name):
    vars_name = name + ".ckpt"
    net_name = name + ".pkl"

    vars_path = os.path.join("./", vars_name)
    net_path = os.path.join("./", net_name)
    
    print("Loading Network")
    with open(net_path, 'rb') as f:
        net = pickle.load(f)
    
    net.load_variables(vars_path)

    return net


