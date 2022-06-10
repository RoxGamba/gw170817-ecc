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
    'ecc_ics'            : 1
    }
    return pardic

def interp_qnt(x, y, x_new):
    f = interpolate.interp1d(x, y, bounds_error=False, fill_value=(0,0))
    yn= f(x_new)
    return yn

# function to compute envelope, taken from stack overflow
def hl_envelopes_idx(s, dmin=1, dmax=1, split=False):
    """
    Input :
    s: 1d-array, data signal from which to extract high and low envelopes
    dmin, dmax: int, optional, size of chunks, use this if the size of the input signal is too big
    split: bool, optional, if True, split the signal in half along its mean, might help to generate the envelope in some cases
    Output :
    lmin,lmax : high/low envelope idx of input signal s
    """

    # locals min      
    lmin = (np.diff(np.sign(np.diff(s))) > 0).nonzero()[0] + 1 
    # locals max
    lmax = (np.diff(np.sign(np.diff(s))) < 0).nonzero()[0] + 1 
    

    if split:
        # s_mid is zero if s centered around x-axis or more generally mean of signal
        s_mid = np.mean(s) 
        # pre-sorting of locals min based on relative position with respect to s_mid 
        lmin = lmin[s[lmin]<s_mid]
        # pre-sorting of local max based on relative position with respect to s_mid 
        lmax = lmax[s[lmax]>s_mid]


    # global max of dmax-chunks of locals max 
    lmin = lmin[[i+np.argmin(s[lmin[i:i+dmin]]) for i in range(0,len(lmin),dmin)]]
    # global min of dmin-chunks of locals min 
    lmax = lmax[[i+np.argmax(s[lmax[i:i+dmax]]) for i in range(0,len(lmax),dmax)]]
    
    return lmin,lmax

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

    def find_merger(self, dynamics=False):
        """
        Find and output merger quantities for eccentric systems 
        based on A22
        """
        from scipy.signal import find_peaks; 

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
            Eb_mrg = interp_qnt(self.dyn['t'], Eb, t_mrg) 
            j_mrg  = interp_qnt(self.dyn['t'], j , t_mrg)
            return t_mrg, A_mrg, p_mrg, omg_mrg, Eb_mrg, j_mrg

        return t_mrg, A_mrg, p_mrg, omg_mrg

    def ecc_omega(self):
        """
        Compute the eccentricity evolution from omega
        """
        _,_, omega_22 = self.load_hlm(2,2)
        omg_a, omg_p  = hl_envelopes_idx(omega_22)        
        omg_p = interp_qnt(self.t[omg_p], omega_22[omg_p], self.t)
        omg_a = interp_qnt(self.t[omg_a], omega_22[omg_a], self.t)
        ecc_omg = (omg_p**(0.5) - omg_a**(0.5))/(omg_p**(0.5)+omg_a**(0.5))
        return ecc_omg, omg_p, omg_a

    def ecc_3PN(self):
        """
        Eq. 4.8 of https://arxiv.org/pdf/1507.07100.pdf
        e_t in Harmonic coordinates
        """

        q  = self.pars['q']
        nu = q/(1+q)**2 
        nu2 = nu*nu
        nu3 = nu2*nu 

        Pi = np.pi
        Pi2 = Pi*Pi
    
        Eb, pph = self.compute_energetics()
        xi  = -Eb*pph**2

        e_0PN  = 1-2*xi
        e_1PN  = -4.-2*nu+ (-1 + 3*nu)*xi
        e_2PN  = (20.-23*nu)/xi -22. + 60*nu + 3*nu2 - (31*nu+4*nu2)*xi
        e_3PN  = (-2016 + (5644 - 123*Pi2)*nu -252*nu2)/(12*xi*xi) + (4848 +(-21128 + 369*Pi2)*nu + 2988*nu2)/(24*xi) - 20 + 298*nu - 186*nu2 - 4*nu3 + (-1*30.*nu + 283./4*nu2 + 5*nu3)*xi
        return np.sqrt(Eb*(Eb*(Eb*e_3PN + e_2PN) + e_1PN) + e_0PN)

    def ecc_radial(self):
        """
        """
        r     = self.dyn['r']
        t     = self.dyn['t']
        rdot    = np.zeros_like(r)        
        rdot[1:]= np.diff(r)/np.diff(t)
        e_r = rdot*r**(0.5)

        _, e_r_env = hl_envelopes_idx(e_r)
        e_r = interp_qnt(t[e_r_env], e_r[e_r_env], self.t)
        return e_r

    def ecc_radial_a(self):
        """
        e_A = r^2*ddot{r}
        https://arxiv.org/pdf/1810.03521.pdf
        """
        r         = self.dyn['r']
        t         = self.dyn['t']
        rdot      = np.zeros_like(r)
        rddot     = np.zeros_like(r)
        rdot[1:]  = np.diff(r)/np.diff(t)
        rddot[1:] = np.diff(rdot)/np.diff(t) 
        e_A       = rddot*r**2

        _, e_A_env = hl_envelopes_idx(e_A)
        e_A = interp_qnt(t[e_A_env], e_A[e_A_env], self.t)
        return e_A