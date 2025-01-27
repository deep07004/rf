from obspy.geodetics import gps2dist_azimuth
import numpy as np
import pygmt.figure
from tqdm import tqdm

def avg_vpvs(mod, d0, d):
    _tmp = mod[mod[:,0]>=d0]
    tmp = _tmp[_tmp[:,0]<=d]
    if tmp.shape[0] == 0:
        return _tmp[0,1],_tmp[0,2]
    elif tmp.shape[0] == 1:
        return tmp[0,1], tmp[0,2]
    vp = 0
    vs = 0
    wt = 0
    for i, a in enumerate(tmp):
        if i > 0:
            dd = tmp[i,0]- tmp[i-1,0]
            wt += dd
            vp += tmp[i-1,1]*dd
            vs += tmp[i-1,2]*dd
    if tmp[-1,0] != d:
        dd = d - tmp[-1,0]
        wt +=dd
        vp += tmp[-1,1]*dd
        vs += tmp[-1,2]*dd
    return vp/wt, vs/wt

class CCPImage(object):
    def __init__(self, profile=None, grid=None, profile_width=100, fz=False, \
                 maxfreq=1.0,model=None, pw_stack=False):
        self.profile = profile
        if not self.profile:
            print("Please define the profile")
            return None
        self.pz = fz
        self.pw = pw_stack
        self.maxfreq = maxfreq
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
        self.width = profile_width
        self.azm = az # profile azimuth
        self.nx = int(np.ceil(self.length/self.grid_w))
        self.ny = int(np.ceil((self.max_depth-self.min_depth)/self.grid_h))
        self.nz = int(np.ceil(self.max_depth/self.grid_w))
        if model:
            self.mod = np.loadtxt(model)
        self.data = np.zeros([self.nx+1,self.ny])
        self.weight = np.zeros([self.nx+1,self.ny])


    def add_data(self, stream=None):
        if not stream or len(stream) == 0:
            print("There is no data to add")
            return None
        for tr in tqdm(stream):
            _data = self.migrate(tr)
            if len(_data) > 0:
                for _tmp in _data:
                    i, j, val, wt = _tmp
                    self.data[i,j] += val
                    self.weight[i,j] += wt


    def migrate(self, tr):
        """
        We will use spherical travel time equation for migration. So eath flattening 
        transfomation is not required and slowness should be in deg/radian.
        """
        P = tr.stats.slowness * 57.295
        baz= tr.stats.back_azimuth
        stla = tr.stats.station_latitude
        stlo = tr.stats.station_longitude
        tshift = tr.stats.onset - tr.stats.starttime
        sampling_rate = tr.stats.sampling_rate
        npts = tr.stats.npts
        out = []
        dist, saz, junk = gps2dist_azimuth(self.profile[0],self.profile[1],stla, stlo)
        dist /=1000.0
        phi = (saz - self.azm )*np.pi/180.0
        xx = np.cos(phi) * dist # distance along the profile
        yy = np.sin(phi) * dist # distance from the profile
        caz = (baz - self.azm)*np.pi/180.0
        if abs(yy) > self.width or (xx < 0 or xx> self.length):
            print("Trace %s is outside of profile diimension. Skipping" %(tr.id))
            return out
        tt = 0.0
        DX = 0.0
        for k in range(self.nz):
            dd0 = self.grid_h * k
            dd = self.grid_h * (k+1)
            vp , vs = avg_vpvs(self.mod, dd0, dd)
            wavep = 0.005*dd+1.0
            wavelength = vs/self.maxfreq
            fz = np.sqrt(wavelength * dd + 0.25 * wavelength * wavelength)
            R = 6371.0 - dd
            # vertical slownesses of P and S wave in spherical coordinate system
            a = np.sqrt(abs((R*R) / (vp * vp) - P * P));
            b = np.sqrt(abs((R*R) / (vs * vs) - P * P));
            q = b - a;
            # time spent in this layer
            dt = ((dd-dd0)/R)*q 
            # horizontal distance covered during this time
            dx = (dd-dd0)* P/b
            DX += dx
            # Find the grid index where the ray segement begins and end
            dx *= np.cos(caz) # project the horizontal progress into the profile
            i = int(np.ceil(xx / self.grid_w))
            j = int(np.ceil((xx + dx) / self.grid_w))
            if j == 0 or j > self.nx:
                # Ray is out of grid boundary
                break
            if i == j: # ray segment start and end in the same cell
                i1 = int(np.ceil((tt + tshift) * sampling_rate))
                if (i1 < 0):
                    i1 = 0
                i2 = int(np.floor((tt + dt + tshift) * sampling_rate))
                if (i2 < 0):
                    i2 = 0
                if (i2 > npts):
                    continue # i.e no sample in cell
                tt += dt # Update time
                i3 = int(np.ceil(fz /self.grid_w))
                data_sum = np.sum(tr.data[i1:i2])
                weight = i2-i1+1
                if i3 < 2:
                    out.append([i,k,data_sum, weight])
                else:
                    for i4 in range(i-i3+1,i+i3):
                        if i4 > 0 and i4 < self.nx:
                            out.append([i4,k,data_sum,weight])
            else:
                nseg = abs(j-i)
                #Compute contribution from each segment
                seg_contri = []
                # Contribution from 1st segment
                if j>i:
                    seg_contri.append((i * self.grid_w - xx)/dx)
                else:
                    seg_contri.append(((i-1) * self.grid_w - xx)/dx)
                # Contributio from segements which cross one complete cell
                for m in range(nseg-2):
                    if j>i:
                        seg_contri.append(self.grid_w/dx)
                    else:
                        seg_contri.append(-self.grid_w/dx)
                # Contribution from the last segment
                if j>i:
                    seg_contri.append(((xx+dx) - (j-1) * self.grid_w )/dx)
                else:
                    seg_contri.append(((xx+dx) - j * self.grid_w )/dx)
                for m in range(nseg):
                    if j>i:
                        ix = i + m
                    else:
                        ix = i - m
                    if ix ==0 or ix > self.nx:
                        break
                    dtx = seg_contri[m]*dt
                    i1 = int(np.ceil((tt + tshift) * sampling_rate))
                    if (i1 < 0):
                        i1 = 0
                    i2 = int(np.floor((tt + dtx + tshift) * sampling_rate))
                    if (i2 < 0):
                        i2 = 0
                    if (i2 > npts):
                        continue # i.e no sample in cell
                    tt += dt # Update time
                    i3 = int(np.ceil(fz /self.grid_w))
                    data_sum = np.sum(tr.data[i1:i2])
                    weight = i2-i1+1
                    if i3 < 2:
                        out.append([i,k,data_sum, weight])
                    else:
                        for i4 in range(i-i3+1,i+i3):
                            if i4 > 0 and i4 < self.nx:
                                out.append([i4,k,data_sum,weight])
            xx += dx
        return out
    def plot(self, vmin = -0.05, vmax = 0.05, fname=None):
        import pygmt
        
        width = 14
        height = ((self.max_depth-self.min_depth)/self.length)*width
        proj = "X%0.2fc/-%0.2fc" %(width, height)
        region = [0, self.length, self.min_depth, self.max_depth]
        data = []
        for i in range(self.nx):
            for j in range(self.ny):
                dist = i*self.grid_w
                depth = j*self.grid_h 
                if self.weight[i,j]>2:
                    _tmp = self.data[i,j]/(self.weight[i,j]**0.75)
                else:
                    _tmp = 0.0
                data.append([dist,depth,_tmp])
        data = np.array(data)
        sp = "%f/%f" %(self.grid_w, self.grid_h)
        grd = pygmt.xyz2grd(data, region=region, spacing=sp)
        fig = pygmt.Figure()
        pygmt.makecpt(cmap="polar",series=[vmin, vmax, 0.001],background=True)
        fig.grdimage(projection=proj,
            grid=grd,
            region=region,
            frame=["af", "WSne" ],
            cmap=True,)
        if fname:
            fig.savefig(fn)
        else:
            fig.show()

