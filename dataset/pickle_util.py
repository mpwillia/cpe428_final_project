
try:
    import cPickle as pickle
except:
    import pickle
import gzip

def dump(obj, path, compress = True):
    
    if compress and not path.endswith('.gz'):
        path = path + '.gz'
    
    if compress:
        with gzip.open(path, 'wb') as f:
            pickle.dump(obj, f)
    else:
        with open(path, 'w') as f:
            pickle.dump(obj, f)

def load(path):
    compress = path.endswith('.gz')
    if compress:
        with gzip.open(path, 'rb') as f:
            return pickle.load(f)
    else:
        with open(path, 'r') as f:
            return pickle.load(f)

