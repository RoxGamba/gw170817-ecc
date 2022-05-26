import numpy as np
"""
Class for EOB Py waveforms; computes various quantities
RG, 05/22
"""

import numpy as np;
from scipy import interpolate
Msuns  = 4.925491025543575903411922162094833998e-6

KMAX = 14

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def mode_to_k(ell, emm):
    return int(ell*(ell-1)/2 + emm-2)

def k_to_ell(k):
    LINDEX = [\
    2,2,\
    3,3,3,\
    4,4,4,4,\
    5,5,5,5,5,\
    6,6,6,6,6,6,\
    7,7,7,7,7,7,7,\
    8,8,8,8,8,8,8,8]
    return LINDEX[k]

def k_to_emm(k):
    MINDEX = [\
    1,2,\
    1,2,3,\
    1,2,3,4,\
    1,2,3,4,5,\
    1,2,3,4,5,6,\
    1,2,3,4,5,6,7,\
    1,2,3,4,5,6,7,8];
    return MINDEX[k]   

def CreateDict(M=1, q=1, chi1z=0., chi2z=0, l1=0, l2=0, ecc=0.1, iota=0, f0=20., srate=4096., interp=1, arg_out=1, ecc_freq=2, use_geom=1):
    """
    Create the dictionary of parameters for Eccentric EOBRunPy
    """

    pardic = {
    'M'                  : 1,
    'q'                  : q,
    'chi1'               : chi1z,
    'chi2'               : chi2z,
    'Lambda1'            : l1,
    'Lambda2'            : l2,
    'distance'           : 1.,
    'initial_frequency'  : f0,
    'ecc'                : ecc,
    'use_geometric_units': use_geom,
    'interp_uniform_grid': interp,
    'domain'             : 0,
    # 'output_dynamics'    : 1,
    'srate_interp'       : srate,
    'inclination'        : iota,
    'output_hpc'         : 0,
    'use_mode_lm'        : [1],      # List of modes to use
    'arg_out'            : arg_out,  # output dynamics and hlm in addition to h+, hx
    'ecc_freq'           : ecc_freq,      #Use periastron (0), average (1) or apastron (2) frequency for initial condition computation. Default = 1
    }
    return pardic

VERBOSE = 0

class Waveform_PyEOB(object):
    """
    Class to analyze EOB simulations
    """

    def __init__(self, params=None):
        import EOBRun_module as EOB
        
        self.pars = params 
        dyn = None
        hlm = None 
        t   = None
        hp  = None
        hc  = None 
        if (VERBOSE):
            print("Run TEOBResumS")

        if 'arg_out' in params.keys():
            if params['arg_out'] == 1:
                t, hp, hc, hlm, dyn = EOB.EOBRunPy(params)
                if (VERBOSE): 
                    print("Done")
            else:
                t, hp, hc = EOB.EOBRunPy(params)
                if (VERBOSE):
                    print("Done")
        else:
            t, hp, hc = EOB.EOBRunPy(params)
            if (VERBOSE):
                print("Done")
        
        self.t  = t 
        self.hp = hp 
        self.hc = hc
        self.hlm= hlm 
        self.dyn= dyn

    def load_hlm(self, ell, emm):
        """
        Load multipolar waveform h_{ell, emm}
        """
        k          = mode_to_k(ell, emm)
        Alm, philm = self.hlm[str(k)][0], self.hlm[str(k)][1]
        omglm      = np.zeros_like(philm)
        omglm[1:]  = np.diff(philm)/np.diff(self.t)/(np.pi*2)

        return Alm, philm, omglm

    def interpolate_dynamics(self, t):
        """
        Interpolate the dynamics at a given time
        """
        dyn = self.dyn
        t_dyn = dyn['t']
        dyn_int = {}
        for k in dyn.keys():
            val = dyn[k]
            fv = interpolate.interp1d(t_dyn, val)
            dyn_int[k] = fv(t)
        return dyn_int

    def compute_energetics(self):
        """
        Compute binding energy Eb and angular momentum
        """
        pars = self.pars
        q    = pars['q']
        q    = float(q)
        nu   =  q/(1.+q)**2
        dyn = self.dyn
        E, j = dyn['E'], dyn['Pphi']
        Eb = (E-1)/nu
        return Eb, j
    
    def cut_at_max_omega(self):
        """
        Cut the waveforms at the maximum of the orbital frequency
        """

        omg = self.dyn['MOmega'] 
        i_cut_dyn = np.argmax(omg)
        t_cut = self.dyn['t'][i_cut_dyn]
        if(self.pars['use_geometric_units'] == 0):
            t_cut = t_cut*self.pars['M']*Msuns

        i_cut_wf = find_nearest(self.t, t_cut)

        self.t  = self.t[:i_cut_wf]
        self.hp = self.hp[:i_cut_wf] 
        self.hc = self.hc[:i_cut_wf] 

        return i_cut_wf

    def find_merger(self, dynamics=False):
        """
        Find and output merger quantities for eccentric systems 
        based on A22
        """
        from scipy.signal import find_peaks; 
        from scipy import interpolate

        A, p, omg = self.load_hlm(2,2)
        
        #find first peak (merger) after umin
        peaks, _ = find_peaks(A, height=0.05)

        # the merger is the last peak of A22
        i_mrg = peaks[-1]

        t_mrg    = self.t[i_mrg]
        A_mrg    = A[i_mrg]
        p_mrg    = p[i_mrg]
        omg_mrg  = omg[i_mrg]
        
        if (dynamics):
            Eb, j = self.compute_energetics()
            fE = interpolate.interp1d(self.dyn['t'], Eb)
            fj = interpolate.interp1d(self.dyn['t'], j)
            Eb_mrg = fE(t_mrg)
            j_mrg  = fj(t_mrg)
            return t_mrg, A_mrg, p_mrg, omg_mrg, Eb_mrg, j_mrg

        return t_mrg, A_mrg, p_mrg, omg_mrg

    def interp_qnt(self, x, y, x_new):
        f = interpolate.interp1d(x, y, bounds_error=False, fill_value=(0,0))
        yn= f(x_new)
        return yn