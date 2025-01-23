from obspy.geodetics import gps2dist_azimuth
import numpy as np

class CCPImage(object):
    def __init__(self, profile=None, grid=None, profile_width=100, fz=False, \
                 model=None, pw_stack=False):
        self.profile = profile
        if not self.profile:
            print("Please define the profile")
            return None
        self.pz = fz
        self.pw = pw_stack
        if not grid:
            self.min_depth = 0
            self.max_depth = 100
            self.grid_h = 5.0
            self.grid_w = 5.0
        else:
            self.min_depth = grid[0]
            self.max_depth = grid[1]
            self.grid_h = grid[2]
            self.grid_w = grid[3]
        dist, az, baz = gps2dist_azimuth(*profile)
        self.length = dist/1000.0 # profile length
        self.azm = az # profile azimuth
        self.nx = int(np.ceil(self.length/self.grid_w))
        self.ny = int(np.ceil((self.max_depth-self.min_depth)/self.grid_h))
        if model:
            mod = np.loadtxt(model)
        self.data = np.zeros([self.nx,self.ny])
        self.weight = np.zeros([self.nx,self.ny])


    def add_data(self, stream=None):
        if not stream or len(stream) == 0:
            print("There is no data to add")
            return self
        
        

