# -*- coding: utf-8 -*-
# Copyright 2013-2019 Tom Eulenfeld, MIT license
"""
Classes and functions for receiver function calculation.
"""
import json
from operator import itemgetter
from pkg_resources import resource_filename
import warnings
from pathlib import Path
from glob import has_magic, glob
from tqdm import tqdm

import numpy as np
from obspy import read, Stream, Trace, UTCDateTime
from obspy.core import AttribDict
from obspy.geodetics import gps2dist_azimuth
from obspy.taup import TauPyModel
from scipy.signal import hilbert
from rf.deconvolve import deconvolve
from rf.harmonics import harmonics
from rf.simple_model import load_model
from rf.util import DEG2KM, _calc_snr, IterMultipleComponents, _add_processing_info


def __get_event_origin_prop(h):
    def wrapper(event):
        try:
            r = (event.preferred_origin() or event.origins[0])[h]
        except IndexError:
            raise ValueError('No origin')
        if r is None:
            raise ValueError('No origin ' + h)
        if h == 'depth':
            r = r / 1000
        return r
    return wrapper


def __get_event_magnitude(event):
    try:
        return (event.preferred_magnitude() or event.magnitudes[0])['mag']
    except IndexError:
        raise ValueError('No magnitude')


def __get_event_id(event):
    evid = event.get('resource_id')
    if evid is not None:
        evid = str(evid)
    return evid


def __SAC2UTC(stats, head):
    from obspy.io.sac.util import get_sac_reftime
    return get_sac_reftime(stats.sac) + stats[head]


def __UTC2SAC(stats, head):
    from obspy.io.sac.util import get_sac_reftime
    return stats[head] - get_sac_reftime(stats.sac)


_STATION_GETTER = (('station_latitude', itemgetter('latitude')),
                   ('station_longitude', itemgetter('longitude')),
                   ('station_elevation', itemgetter('elevation')))
_EVENT_GETTER = (
    ('event_latitude', __get_event_origin_prop('latitude')),
    ('event_longitude', __get_event_origin_prop('longitude')),
    ('event_depth', __get_event_origin_prop('depth')),
    ('event_magnitude', __get_event_magnitude),
    ('event_time', __get_event_origin_prop('time')),
    ('event_id', __get_event_id))

# header values which will be written to waveform formats (SAC and Q)
# H5 simply writes all stats entries
_HEADERS = (tuple(zip(*_STATION_GETTER))[0] +
            tuple(zip(*_EVENT_GETTER))[0][:-1] + (  # do not write event_id
            'onset', 'type', 'phase', 'moveout',
            'distance', 'back_azimuth', 'inclination', 'slowness',
            'pp_latitude', 'pp_longitude', 'pp_depth',
            'box_pos', 'box_length','snr'))

# The corresponding header fields in the format
# The following headers can at the moment only be stored for H5:
# slowness_before_moveout, box_lonlat, event_id
_FORMATHEADERS = {'sac': ('stla', 'stlo', 'stel', 'evla', 'evlo',
                          'evdp', 'mag', 'o', 'a',
                          'kuser0', 'kuser1', 'kuser2',
                          'gcarc', 'baz', 'user0', 'user1',
                          'user2', 'user3', 'user4',
                          'user5', 'user6', 'user7'),
                  # field 'COMMENT' is violated for different information
                  'sh': ('COMMENT', 'COMMENT', 'COMMENT',
                         'LAT', 'LON', 'DEPTH',
                         'MAGNITUDE', 'ORIGIN', 'P-ONSET',
                         'COMMENT', 'COMMENT', 'COMMENT',
                         'DISTANCE', 'AZIMUTH', 'INCI', 'SLOWNESS',
                         'COMMENT', 'COMMENT', 'COMMENT',
                         'COMMENT', 'COMMENT','COMMENT')}
_HEADER_CONVERSIONS = {'sac': {'onset': (__SAC2UTC, __UTC2SAC),
                               'event_time': (__SAC2UTC, __UTC2SAC)}}


_TF = '.datetime:%Y-%m-%dT%H:%M:%S'

_H5INDEX = {
    'rf': ('waveforms/{network}.{station}.{location}/{event_time%s}/' % _TF +
           '{channel}_{starttime%s}_{endtime%s}' % (_TF, _TF)),
    'profile': 'waveforms/{channel[2]}_{box_pos}'
}

def _to_rf_from_ah(stats):
    ah = stats.ah
    stats.event_latitude = ah.event.latitude
    stats.event_longitude = ah.event.longitude
    stats.event_depth = ah.event.depth
    stats.event_magnitude = ah.extras[0]
    stats.event_time = ah.event.origin_time
    stats.station_elevation = ah.station.elevation
    stats.station_latitude = ah.station.latitude
    stats.station_longitude = ah.station.longitude
    return stats

def read_rf(pathname_or_url=None, format=None, **kwargs):
    """
    Read waveform files into RFStream object.

    See :func:`~obspy.core.stream.read` in ObsPy.
    """
    stream = Stream()
    if pathname_or_url is None:   # use example file
        fname = resource_filename('rf', 'example/minimal_example.tar.gz')
        pathname_or_url = fname
        format = 'SAC'
    if Path(pathname_or_url).is_file():
        st = read(pathname_or_url, format=format, **kwargs)
        for tr in st:
                stream.append(tr)
    if has_magic(pathname_or_url):
        ff = glob(pathname_or_url)      
        print("\nLoading %d traces for processing ..." %(len(ff)))
        for f in tqdm(ff):
            try:
                st = read(f, format=format, **kwargs)
            except Exception as e:
                print(e)
            for tr in st:
                stream.append(tr)
    return RFStream(stream)


class RFStream(Stream):

    """
    Class providing the necessary functions for receiver function calculation.

    :param traces: list of traces, single trace or stream object

    To initialize a RFStream from a Stream object use

    >>> rfstream = RFStream(stream)

    To initialize a RFStream from a file use

    >>> rfstream = read_rf('test.SAC')

    Format specific headers are loaded into the stats object of all traces.
    """

    def __init__(self, traces=None):
        self.traces = []
        if isinstance(traces, Trace):
            traces = [traces]
        if traces:
            for tr in traces:
                if not isinstance(tr, RFTrace):
                    tr = RFTrace(trace=tr)
                self.traces.append(tr)

    def __is_set(self, header):
        return all(header in tr.stats for tr in self)

    def __get_unique_header(self, header):
        values = set(tr.stats[header] for tr in self if header in tr.stats)
        if len(values) > 1:
            warnings.warn('Header %s has different values in stream.' % header)
        if len(values) == 1:
            return values.pop()

    @property
    def type(self):
        """Property for the type of stream, 'rf', 'profile' or None"""
        return self.__get_unique_header('type')

    @type.setter
    def type(self, value):
        for tr in self:
            tr.stats.type = value

    @property
    def method(self):
        """Property for used rf method, 'P' or 'S'"""
        phase = self.__get_unique_header('phase')
        if phase is not None:
            return phase[-1].upper()

    @method.setter
    def method(self, value):
        for tr in self:
            tr.stats.phase = value

    def write(self, filename, format, sh_compat=False, **kwargs):
        """
        Save stream to file including format specific headers.

        See `Stream.write() <obspy.core.stream.Stream.write>` in ObsPy.

        :param sh_compat: Ensure files in Q format can be read with
            SeismicHandler (default: False). If set to True the COMMENT field
            will be deleted which might result in loss of some metadata.
            (see issue #37).
        """
        if len(self) == 0:
            return
        for tr in self:
            tr._write_format_specific_header(format, sh_compat=sh_compat)
            if format.upper() == 'Q':
                tr.stats.station = tr.id
        if format.upper() == 'H5':
            index = self.type
            if index is None and 'event_time' in self[0].stats:
                index = 'rf'
            if index:
                import obspyh5
                old_index = obspyh5._INDEX
                obspyh5.set_index(_H5INDEX[index])
        super(RFStream, self).write(filename, format, **kwargs)
        if format.upper() == 'H5' and index:
            obspyh5.set_index(old_index)
        if format.upper() == 'Q':
            for tr in self:
                tr.stats.station = tr.stats.station.split('.')[1]

    def trim2(self, starttime=None, endtime=None, reftime=None, **kwargs):
        """
        Alternative trim method accepting relative times.

        See :meth:`~obspy.core.stream.Stream.trim`.

        :param starttime,endtime: accept UTCDateTime or seconds relative to
            reftime
        :param reftime: reference time, can be an UTCDateTime object or a
            string. The string will be looked up in the stats dictionary
            (e.g. 'starttime', 'endtime', 'onset').
        """
        for tr in self.traces:
            t1 = tr._seconds2utc(starttime, reftime=reftime)
            t2 = tr._seconds2utc(endtime, reftime=reftime)
            tr.trim(t1, t2, **kwargs)
        self.traces = [_i for _i in self.traces if (_i.stats.endtime - _i.stats.starttime) == (endtime - starttime)]
        return self

    def slice2(self, starttime=None, endtime=None, reftime=None,
               keep_empty_traces=False, **kwargs):
        """
        Alternative slice method accepting relative times.

        See :meth:`~obspy.core.stream.Stream.slice` and `trim2()`.
        """
        traces = []
        for tr in self:
            t1 = tr._seconds2utc(starttime, reftime=reftime)
            t2 = tr._seconds2utc(endtime, reftime=reftime)
            sliced_trace = tr.slice(t1, t2, **kwargs)
            if not keep_empty_traces and not sliced_trace.stats.npts:
                continue
            traces.append(sliced_trace)
        return self.__class__(traces)
    def rotate_by_nsv(self, vp=None, vs=None):
        """
        Roate using Kennet's rotation matrix
        """
        if vp is None:
            vp = 5.8
        if vs is None:
            vs = 2.75
        if len(self) != 3:
            print("All three components required for rotation by NSV. %d provided" %(len(self)))
            return self
        try:
            self.rotate('NE->RT')
        except Exception as e:
            print(e)
            return self
        ray_par = self[0].stats.slowness/111.1949
        qalpha = np.sqrt(1/vp**2 -ray_par**2)
        qbeta = np.sqrt(1/vs**2 -ray_par**2)
        z = self.select(component="Z")[0].data*-1.0 # Z need to be positive down. So flip it.
        r = self.select(component="R")[0].data
        t = self.select(component="T")[0].data
        P = (ray_par*(vs)**2/vp) * r + (((vs*ray_par)**2 -0.5)/(vp*qalpha)) * z
        SV = ((0.5 - (vs*ray_par)**2)/(vs * qbeta)) * r + ray_par*vs * z
        SH = t/2.0
        self[0].stats.channel = self[0].stats.channel[0:2]+"L"
        self[0].data = P
        self[1].stats.channel = self[1].stats.channel[0:2]+"Q"
        self[1].data = SV
        self[2].stats.channel = self[2].stats.channel[0:2]+"T"
        self[2].data = SH
        return self


    def calc_snr(self,phase="P"):
        import concurrent.futures
        def iter3c(stream):
            return IterMultipleComponents(stream, key='onset',
                                          number_components=(3))
        processes = []
        traces = []
        print("\n Caluclating %s SNR\n"%(phase))
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for stream3c in iter3c(self):
                if len(stream3c) ==3:
                    p = executor.submit(_calc_snr,stream3c, phase)
                    processes.append(p)
            for f in tqdm(concurrent.futures.as_completed(processes), total=len(processes)):
                if f.result() is not None and isinstance(f.result(), RFStream):
                    for tr in f.result():
                        traces.append(tr)
        self.traces = traces
        return self
    
    def deconvolve(self, *args, **kwargs):
        """
        Deconvolve source component of stream.

        All args and kwargs are passed to the function
        `~rf.deconvolve.deconvolve()`.
        """
        rsp = deconvolve(self, *args, **kwargs)
        self.traces = rsp.traces
        return self

    @_add_processing_info
    def rf(self, method=None, filter=None, trim=None, downsample=None,
           rotate='ZNE->LQT',vp=None, vs=None, deconvolve='time', source_components=None,
           **kwargs):
        """
        Calculate receiver functions in-place.

        :param method: 'P' for P receiver functions, 'S' for S receiver
            functions, if None method will be determined from the phase
        :param dict filter: filter stream with its
            `~obspy.core.stream.Stream.filter` method and given kwargs
        :type trim: tuple (start, end)
        :param trim: trim stream relative to P- or S-onset
             with `trim2()` (seconds)
        :param float downsample: downsample stream with its
            :meth:`~obspy.core.stream.Stream.decimate` method to the given
            frequency
        :type rotate: 'ZNE->LQT' or 'NE->RT'
        :param rotate: rotate stream with its
            :meth:`~obspy.core.stream.Stream.rotate`
            method with the angles given by the back_azimuth and inclination
            attributes of the traces stats objects. You can set these to your
            needs or let them be computed by :func:`~rf.rfstream.rfstats`.
        :param deconvolve: 'time', 'waterlevel', 'iterative' or 'multitaper'
            for time domain damped, frequency domain water level,
            time domain iterative, or frequency domain multitaper
            deconvolution using the stream's `deconvolve()` method.
            See `~.deconvolve.deconvolve()`,
            `.deconv_time()`, `.deconv_waterlevel()`,
            `.deconv_iterative()`, and `.deconv_multitaper()`
            for further documentation.
        :param source_components: parameter is passed to deconvolve.
            If None, source components will be chosen depending on method.
        :param \*\*kwargs: all other kwargs not mentioned here are
            passed to deconvolve

        After performing the deconvolution the Q/R and T components are
        multiplied by -1 to get a positive phase for a Moho-like positive
        velocity contrast. Furthermore for method='S' all components are
        mirrored at t=0 for a better comparison with P receiver functions.
        See source code of this function for the default
        deconvolution windows.
        """
        #print("Deconvolution parameters\n")
        #print("\n",deconvolve, source_components,"\n")
        import concurrent.futures
        def iter3c(stream):
            return IterMultipleComponents(stream, key='onset',
                                          number_components=(2, 3))

        if method is None:
            method = self.method
        if method is None or method not in 'PS':
            msg = "method must be one of 'P', 'S', but is '%s'"
            raise ValueError(msg % method)
        if source_components is None:
            source_components = 'LZ' if method == 'P' else 'QR'
        if filter:
            self.filter(**filter)
        if trim:
            self.trim2(*trim, reftime='onset')
        if downsample:
            for tr in self:
                if downsample <= tr.stats.sampling_rate:
                    tr.decimate(int(tr.stats.sampling_rate) // downsample)
        if rotate:
            print("Rotating %s" %rotate)
            for stream3c in tqdm(iter3c(self)):
                try:
                    npts = [tr.stats.npts for tr in stream3c]
                    npts.sort()
                    if npts[0] != npts[-1] or len(stream3c) !=3:
                        for tr in stream3c:
                            self.remove(tr)
                    else:
                        if rotate.upper() == "NSV":
                            stream3c.rotate_by_nsv(vp, vs)
                        else:
                            stream3c.rotate(rotate)
                except Exception as e:
                    print(e)
                
        # Multiply -1 on Q component, because Q component is pointing
        # towards the event after the rotation with ObsPy.
        # For a positive phase at a Moho-like velocity contrast,
        # the Q component has to point away from the event.
        # This is not necessary for the R component which points already
        # away from the event.
        # (compare issue #4)
        for tr in self:
            if not rotate.upper()=="NSV" and tr.stats.channel.endswith('Q'):
                tr.data = -tr.data
        if deconvolve:
            #for stream3c in iter3c(self):
            #    kwargs.setdefault('winsrc', method)
            #    stream3c.deconvolve(method=deconvolve,
            #                        source_components=source_components,
            #                        **kwargs)
            print("\nPerforming deconvolution ...")
            processes = []
            traces = []
            with concurrent.futures.ProcessPoolExecutor() as executor:
                for stream3c in iter3c(self):
                    src = [tr for tr in stream3c if tr.stats.channel[-1] in source_components]
                    kwargs.setdefault('winsrc', method)
                    if len(stream3c) ==3 and len(src)==1:
                        p = executor.submit(stream3c.deconvolve, method=deconvolve,
                                            source_components=source_components, **kwargs)
                        processes.append(p)
                    else:
                        for _tr in stream3c:
                            print(_tr.stats.starttime, _tr.stats.endtime)
                for f in tqdm(concurrent.futures.as_completed(processes), total=len(processes)):
                    if f.result() is not None and isinstance(f.result(), RFStream):
                        for tr in f.result():
                            traces.append(tr)
            self.traces = traces
                
        # Mirrow Q/R and T component at 0s for S-receiver method for a better
        # comparison with P-receiver method (converted Sp wave arrives before
        # S wave, but converted Ps wave arrives after P wave)
        if method == 'S':
            for tr in self:
                tr.data = tr.data[::-1]
                tr.stats.onset = tr.stats.starttime + (tr.stats.endtime -
                                                       tr.stats.onset)
        self.type = 'rf'
        if self.method != method:
            self.method = method
        return self

    @_add_processing_info
    def moveout(self, phase=None, ref=6.4, model='iasp91'):
        """
        In-place moveout correction to a reference slowness.

        Needs stats attributes slowness and onset.

        :param phase: 'Ps', 'Sp', 'Ppss' or other multiples, if None is set to
            'Ps' for P receiver functions or 'Sp' for S receiver functions
        :param ref: reference ray parameter in s/deg
        :param model: Path to model file
            (see `.SimpleModel`, default: iasp91)
        """
        if phase is None:
            phase = self.method + {'P': 's', 'S': 'p'}[self.method]
        model = load_model(model)
        print("\nMoveout correcting for phase %s and reference slowness %0.2f\n" %(phase, ref))
        model.moveout(self, phase=phase, ref=ref)
        for tr in self:
            tr.stats.moveout = phase
            tr.stats.slowness_after_moveout = ref
        return self

    def ppoints(self, pp_depth, pp_phase=None, model='iasp91'):
        """
        Return coordinates of piercing point calculated by 1D ray tracing.

        Piercing point coordinates are stored in the
        stats attributes plat and plon. Needs stats attributes
        station_latitude, station_longitude, slowness and back_azimuth.

        :param pp_depth: depth of interface in km
        :param pp_phase: 'P' for piercing points of P wave, 'S' for piercing
            points of S wave or multiples, if None will be
            set to 'S' for P receiver functions or 'P' for S receiver functions
        :param model: path to model file (see `.SimpleModel`, default: iasp91)
        :return: NumPy array with coordinates of piercing points
        """
        if pp_phase is None:
            pp_phase = {'P': 'S', 'S': 'P'}[self.method]
        model = load_model(model)
        for tr in self:
            model.ppoint(tr.stats, pp_depth, phase=pp_phase)
        return np.array([(tr.stats.pp_latitude, tr.stats.pp_longitude)
                         for tr in self])

    @_add_processing_info
    def stack(self, pws=False):
        """
        Return stack of traces with same seed ids.

        Traces with same id need to have the same number of datapoints.
        Each trace in the returned stream will correspond to one unique seed
        id.
        """
        ids = set(tr.id for tr in self)
        tr = self[0]
        traces = []
        for id in ids:
            data = np.zeros(self[0].stats.npts)
            weight = np.zeros(self[0].stats.npts, dtype=complex)
            net, sta, loc, cha = id.split('.')
            nb = 0
            for tr in self:
                if tr.id == id:
                    nb+=1
                    data += tr.data
                    if pws:
                        hilb = hilbert(tr.data)
                        phase = np.arctan2(hilb.imag, hilb.real)
                        weight += np.exp(1j*phase)
                    
            data /= float(nb)
            weight = np.real(abs(weight/float(nb)))
            if pws:
                data *= weight
            header = {'network': net, 'station': sta, 'location': loc,
                      'channel': cha, 'sampling_rate': tr.stats.sampling_rate}
            for entry in ('phase', 'moveout', 'station_latitude',
                          'station_longitude', 'station_elevation',
                          'processing'):
                if entry in tr.stats:
                    header[entry] = tr.stats[entry]
            tr2 = RFTrace(data=data, header=header)
            if 'onset' in tr.stats:
                onset = tr.stats.onset - tr.stats.starttime
                tr2.stats.onset = tr2.stats.starttime + onset
            tr2.stats.nbin = nb
            traces.append(tr2)
        return self.__class__(traces)
    
    
    def bin(self, key=None, start=None, stop=None, nbins=None, pc_overlap=None, pws=False, phase=None, ref=6.4):
        traces = []
        if key is None:
            key="slowness"
        if nbins is None:
            nbins=36
        if key not in ['slowness', 'baz', 'dist']:
            print("\n\nBinning by %s is not implemented.\n\n" %key)
        if pc_overlap is None:
            pc_overlap = 0
        if key == 'slowness':
            stats = [tr.stats.slowness for tr in self]
        if key == 'baz':
            stats = [tr.stats.back_azimuth for tr in self]
        if key == 'dist':
            stats = [tr.stats.distance for tr in self]
        stats.sort()
        if start is None:
            start = stats[0]
        if stop is None:
            stop = stats[-1]
        bins = np.linspace(start, stop, nbins+1)
        for i in range(nbins):
            lh = bins[i]
            rh = bins[i+1]+pc_overlap*(bins[i+1]-bins[i])
            tmp = self.extract(key,lh,rh).copy()
            print("Bin: %0.2f - %0.2f has %d RFs" %(lh,rh,len(tmp)))
            if len(tmp) ==0:
                continue
            tmp = tmp.stack(pws)
            if key == 'baz':
                for tr in tmp:
                    tr.stats.back_azimuth = bins[i]+(bins[i+1]-bins[i])/2
                    traces.append(tr)
            if key == 'dist':
                for tr in tmp:
                    tr.stats.distance = bins[i]+(bins[i+1]-bins[i])/2
                    traces.append(tr)
            if key == 'slowness':
                for tr in tmp:
                    tr.stats.slowness = bins[i]+(bins[i+1]-bins[i])/2
                    traces.append(tr)
        return RFStream(traces)
    

    def harmonics(self, *args, **kwargs):
        """
        Perform harmonic decomposition on stream.

        All args and kwargs are passed to the function
        `~rf.harmonics.harmonics()`.
        """
        hd = harmonics(self, *args, **kwargs)
        return hd

    def profile(self, *args, **kwargs):
        """
        Return profile of receiver functions in the stream.

        See `.profile.profile()` for help on arguments.
        """
        from rf.profile import profile
        return profile(self, *args, **kwargs)
    def extract(self, key=None, min_val=None, max_val=None):
        """
        Extract traces based on header variable and it's min max value.
        """
        stream = RFStream()
        traces = []
        st = self.copy()
        if key.upper() not in ['SLOWNESS','SNR','BAZ', 'DIST','ONSET']:
            print("Extracting by %s is not yet implemented" %key)
            return stream
        if key.upper() == "SLOWNESS":
            if min_val is not None:
                min_slo = float(min_val)
            else:
                min_slo = 4.5
            if max_val is not None:
                max_slo = float(max_val)
            else:
                max_slo = 13.0
            val = np.array([tr.stats.slowness for tr in st])
            ii = np.where((val>=min_slo) & (val<=max_slo))[0]
            for i in ii:
                traces.append(st[i])
        if key.upper() == "SNR":
            if min_val is not None:
                min_snr = float(min_val)
            else:
                min_snr = 0.0
            if max_val is not None:
                max_snr = float(max_val)
            else:
                max_snr = 1000.0
            val = np.array([tr.stats.snr for tr in st])
            ii = np.where((val>=min_snr) & (val<=max_snr))[0]
            for i in ii:
                traces.append(st[i])
        if key.upper() == "BAZ":
            if min_val is not None:
                min_baz = float(min_val)
            else:
                min_baz = 0.0
            if max_val is not None:
                max_baz = float(max_val)
            else:
                max_baz = 360.0
            val = np.array([tr.stats.back_azimuth for tr in st])
            ii = np.where((val>=min_baz) & (val<=max_baz))[0]
            for i in ii:
                traces.append(st[i])
        if key.upper() == "DIST":
            if min_val is not None:
                min_dist = float(min_val)
            else:
                min_dist = 0.0
            if max_val is not None:
                max_dist = float(max_val)
            else:
                max_dist = 360.0
            val = np.array([tr.stats.distance for tr in st])
            ii = np.where((val>=min_dist) & (val<=max_dist))[0]
            for i in ii:
                traces.append(st[i])
        if key.upper() == "ONSET":
            if not isinstance(min_val, UTCDateTime):
                try:
                    min_onset = UTCDateTime(min_val)
                except Exception as e:
                    print(e)
                    return None
            else:
                min_onset = min_val
            if not isinstance(max_val, UTCDateTime):
                try:
                    max_onset = UTCDateTime(max_val)
                except Exception as e:
                    print(e)
                    return None
            else:
                max_onset = max_val
            for tr in st:
                if tr.stats.onset >= min_onset and tr.stats.onset <= max_onset:
                    traces.append(tr)

        stream.traces = traces
        return stream
            


    def plot_rf(self, *args, **kwargs):
        """
        Create receiver function plot.

        See `.imaging.plot_rf()` for help on arguments.
        """
        from rf.imaging import plot_rf
        return plot_rf(self, *args, **kwargs)

    def plot_profile(self, *args, **kwargs):
        """
        Create receiver function profile plot.

        See `.imaging.plot_profile()` for help on arguments.
        """
        from rf.imaging import plot_profile
        return plot_profile(self, *args, **kwargs)


class RFTrace(Trace):

    """
    Class providing the Trace object for receiver function calculation.
    """

    def __init__(self, data=np.array([]), header=None, trace=None):
        if header is None:
            header = {}
        if trace is not None:
            data = trace.data
            header = trace.stats
        super(RFTrace, self).__init__(data=data, header=header)
        st = self.stats
        if ('_format'in st and st._format.upper() == 'Q' and
                st.station.count('.') > 1):
            st.network, st.station, st.location = st.station.split('.')[:3]
        self._read_format_specific_header()

    def __str__(self, id_length=None):
        if 'onset' not in self.stats:
            return super(RFTrace, self).__str__(id_length=id_length)
        out = []
        type_ = self.stats.get('type')
        if type_ is not None:
            m = self.stats.get('phase')
            m = m[-1].upper() if m is not None else ''
            o1 = m + 'rf'
            if type_ != 'rf':
                o1 = o1 + ' ' + type_
            if self.id.startswith('...'):
                o1 = o1 + ' (%s)' % self.id[-1]
            else:
                o1 = o1 + ' ' + self.id
        else:
            o1 = self.id
        out.append(o1)
        t1 = self.stats.starttime - self.stats.onset
        t2 = self.stats.endtime - self.stats.onset
        o2 = '%.1fs - %.1fs' % (t1, t2)
        if self.stats.starttime.timestamp != 0:
            o2 = o2 + ' onset:%s' % self.stats.onset
        out.append(o2)
        out.append('{sampling_rate} Hz, {npts} samples')
        o3 = []
        if 'event_magnitude' in self.stats:
            o3.append('mag:{event_magnitude:.1f}')
        if 'distance' in self.stats:
            o3.append('dist:{distance:.1f}')
        if'back_azimuth' in self.stats:
            o3.append('baz:{back_azimuth:.1f}')
        if 'box_pos' in self.stats:
            o3.append('pos:{box_pos:.2f}km')
        if 'slowness' in self.stats:
            o3.append('slow:{slowness:.2f}')
        if 'moveout' in self.stats:
            o3.append('({moveout} moveout)')
        if np.ma.count_masked(self.data):
            o3.append('(masked)')
        out.append(' '.join(o3))
        return ' | '.join(out).format(**self.stats)

    def _read_format_specific_header(self, format=None):
        st = self.stats
        if format is None:
            if '_format' not in st:
                return
            format = st._format
        format = format.lower()
        if format == "ah":
            st = _to_rf_from_ah(st)
            return
        if format == 'q':
            format = 'sh'
        try:
            header_map = zip(_HEADERS, _FORMATHEADERS[format])
        except KeyError:
            # file format is H5 or not supported
            return
        for head, head_format in header_map:
            try:
                value = st[format][head_format]
            except KeyError:
                continue
            else:
                if format == 'sac' and '-12345' in str(value):
                    pass
                elif format == 'sh' and head_format == 'COMMENT':
                    try:
                        st.update(json.loads(value))
                    except json.JSONDecodeError:
                        pass
                    continue
                else:
                    st[head] = value
            try:
                convert = _HEADER_CONVERSIONS[format][head][0]
                st[head] = convert(st, head)
            except KeyError:
                pass

    def _write_format_specific_header(self, format, sh_compat=False):
        st = self.stats
        format = format.lower()
        if format == 'q':
            format = 'sh'
        elif format == 'sac':
            # make sure SAC reference time is set
            from obspy.io.sac.util import obspy_to_sac_header
            self.stats.sac = obspy_to_sac_header(self.stats)
        try:
            header_map = zip(_HEADERS, _FORMATHEADERS[format])
        except KeyError:
            # file format is H5 or not supported
            return
        if format not in st:
            st[format] = AttribDict({})
        if format == 'sh':
            comment = {}
        for head, head_format in header_map:
            if format == 'sh' and head_format == 'COMMENT':
                try:
                    comment[head] = st[head]
                except KeyError:
                    pass
                continue
            try:
                val = st[head]
            except KeyError:
                continue
            try:
                convert = _HEADER_CONVERSIONS[format][head][1]
                val = convert(st, head)
            except KeyError:
                pass
            st[format][head_format] = val
        if format == 'sh':
            if sh_compat:
                st[format].pop('COMMENT', None)
            elif len(comment) > 0:
                def default(obj):  # convert numpy types
                    return np.asarray(obj).item()
                st[format]['COMMENT'] = json.dumps(
                    comment, separators=(',', ':'), default=default)

    def _seconds2utc(self, seconds, reftime=None):
        """Return UTCDateTime given as seconds relative to reftime"""
        from collections.abc import Iterable
        from obspy import UTCDateTime as UTC
        if isinstance(seconds, Iterable):
            return [self._seconds2utc(s, reftime=reftime) for s in seconds]
        if isinstance(seconds, UTC) or reftime is None or seconds is None:
            return seconds
        if not isinstance(reftime, UTC):
            reftime = self.stats[reftime]
        return reftime + seconds

    def write(self, filename, format, **kwargs):
        """
        Save current trace into a file  including format specific headers.

        See `Trace.write() <obspy.core.trace.Trace.write>` in ObsPy.
        """
        RFStream([self]).write(filename, format, **kwargs)


def obj2stats(event=None, station=None):
    """
    Map event and station object to stats with attributes.

    :param event: ObsPy `~obspy.core.event.event.Event` object
    :param station: station object with attributes latitude, longitude and
        elevation
    :return: ``stats`` object with station and event attributes
    """
    stats = AttribDict({})
    if event is not None:
        for key, getter in _EVENT_GETTER:
            stats[key] = getter(event)
    if station is not None:
        for key, getter in _STATION_GETTER:
            stats[key] = getter(station)
    return stats


def rfstats(obj=None, event=None, station=None,
            phase='P', dist_range='default', tt_model='iasp91',
            pp_depth=None, pp_phase=None, model='iasp91'):
    """
    Calculate ray specific values like slowness for given event and station.

    :param obj: `~obspy.core.trace.Stats` object with event and/or station
        attributes. Can be None if both event and station are given.
        It is possible to specify a stream object, too. Then, rfstats will be
        called for each Trace.stats object and traces outside dist_range will
        be discarded.
    :param event: ObsPy `~obspy.core.event.event.Event` object
    :param station: dictionary like object with items latitude, longitude and
        elevation
    :param phase: string with phase. Usually this will be 'P' or
        'S' for P and S receiver functions, respectively.
    :type dist_range: tuple of length 2
    :param dist_range: if epicentral of event is not in this intervall, None
        is returned by this function,\n
        if phase == 'P' defaults to (30, 90),\n
        if phase == 'S' defaults to (50, 85),\n
        if phase == 'SKS' defaults to (85, 120)
    :param tt_model: model for travel time calculation.
        (see the `obspy.taup` module, default: iasp91)
    :param pp_depth: Depth for piercing point calculation
        (in km, default: None -> No calculation)
    :param pp_phase: Phase for pp calculation (default: 'S' for P-receiver
        function and 'P' for S-receiver function)
    :param model: Path to model file for pp calculation
        (see `.SimpleModel`, default: iasp91)
    :return: `~obspy.core.trace.Stats` object with event and station
        attributes, distance, back_azimuth, inclination, onset and
        slowness or None if epicentral distance is not in the given interval.
        Stream instance if stream was specified instead of stats.
    """
    import concurrent.futures
    if isinstance(obj, (Stream, RFStream)):
        stream = obj
        kwargs = {'event': event, 'station': station,
                  'phase': phase, 'dist_range': dist_range,
                  'tt_model': tt_model, 'pp_depth': pp_depth,
                  'pp_phase': pp_phase, 'model': model}
        traces = []
        processes =[]
        print("Calculating important ray specific parameters for receiver function calculation...")
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for tr in stream:
                p = executor.submit(rfstats, tr, event, station, phase, dist_range,
                                    tt_model, pp_depth, pp_phase, model)
                processes.append(p)
            for f in tqdm(concurrent.futures.as_completed(processes), total=len(processes)):
                if f.result() is not None:
                    traces.append(f.result())
        stream.traces = traces
        return stream
    if dist_range == 'default':
        if phase.upper() == 'P':
            dist_range = (30, 90)
        elif phase.upper() == 'S':
            dist_range = (50, 85)
        elif phase.upper() == "SKS":
            dist_range = (85, 120)
        else:
            raise ValueError('Please specify dist_range parameter')
    st = AttribDict({}) if obj is None else obj
    if event is not None and station is not None:
        st.stats.update(obj2stats(event=event, station=station))
    dist, baz, _ = gps2dist_azimuth(st.stats.station_latitude,
                                    st.stats.station_longitude,
                                    st.stats.event_latitude,
                                    st.stats.event_longitude)
    dist = dist / 1000 / DEG2KM
    if dist_range and not dist_range[0] <= dist <= dist_range[1]:
        return
    tt_model = TauPyModel(model=tt_model)
    arrivals = tt_model.get_travel_times(st.stats.event_depth, dist, (phase,))
    if len(arrivals) == 0:
        raise Exception('TauPy does not return phase %s at distance %s' %
                        (phase, dist))
    if len(arrivals) > 1:
        msg = ('TauPy returns more than one arrival for phase %s at '
               'distance %s -> take first arrival')
        warnings.warn(msg % (phase, dist))
    arrival = arrivals[0]
    onset = st.stats.event_time + arrival.time
    inc = arrival.incident_angle
    slowness = arrival.ray_param_sec_degree
    st.stats.update({'distance': dist, 'back_azimuth': baz, 'inclination': inc,
                  'onset': onset, 'slowness': slowness, 'phase': phase})
    if pp_depth is not None:
        model = load_model(model)
        if pp_phase is None:
            pp_phase = phase.upper()[-1]
        model.ppoint(st.stats, pp_depth, phase=pp_phase)
    return st
