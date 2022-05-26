import numpy as np; import matplotlib.pyplot as plt
import eob_waveform as eobwf

def GenLALWfFD(pars, approx='TaylorF2Ecc'):
    try:
        import lalsimulation as lalsim
        import lal  
    except Exception:
        print("Error importing LALSimulation and/or LAL")

    # create empty LALdict
    params = lal.CreateDict()
    
    # read in from pars
    q    = pars['q']
    M    = pars['M']
    DL   = pars['distance']*1e6*lal.PC_SI
    iota = pars['inclination']
    c1z  = pars['chi1']
    c2z  = pars['chi2']
    df   = pars['df']
    flow = pars['initial_frequency']
    srate= pars['srate_interp'] 
    lam1 = pars['Lambda1']
    lam2 = pars['Lambda2']
    ecc  = pars['ecc']
    mean_anomaly = 0.
    lalsim.SimInspiralWaveformParamsInsertTidalLambda1(params, lam1)
    lalsim.SimInspiralWaveformParamsInsertTidalLambda2(params, lam2)

    # Spin, Tidal and Phase orders (3, 6 and 3.5 PN)
    lalsim.SimInspiralWaveformParamsInsertPNSpinOrder(params , 6)
    lalsim.SimInspiralWaveformParamsInsertPNTidalOrder(params, 12)
    lalsim.SimInspiralWaveformParamsInsertPNPhaseOrder(params, 7)   
    lalsim.SimInspiralWaveformParamsInsertPNEccentricityOrder(params, 4)   

    # Compute masses
    m1   = M*q/(1.+q)
    m2   = M/(1.+q)
    m1SI = m1*lal.MSUN_SI
    m2SI = m2*lal.MSUN_SI

    app  = lalsim.GetApproximantFromString(approx)
    hpf, hcf = lalsim.SimInspiralFD(m1SI,m2SI,0.,0.,c1z,0.,0.,c2z,DL,iota,0.,0.,ecc,mean_anomaly,df,flow,srate/2,flow,params,app)
    f        = np.array(range(0, len(hpf.data.data)))*hpf.deltaF
    return f, hpf.data.data, hcf.data.data

if __name__ == "__main__":
    # compare various phases of TF2

    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()

    for ecc in [0., 0.1, 0.2, 0.6]:

        pars = eobwf.CreateDict(M=2.7, l1 = 0, l2 = 0, ecc=ecc, use_geom=0)
        pars['df'] = 1./(8*128)
        f, hpf, hcf = GenLALWfFD(pars)
        pp = np.unwrap(np.angle(hpf)) 
        ax1.semilogx(f, pp, label="ecc = "+str(ecc))
        ax2.loglog(f, abs(hpf), label="ecc = "+str(ecc))

    ax1.set_xlabel('f')
    ax1.set_ylabel('$\Psi(f)$')
    ax2.set_xlabel('f')
    ax2.set_ylabel('$|h_+|(f)$')
    ax1.legend()
    ax2.legend()
    fig1.savefig('../figs/taylorf2ecc_phase.png')
    fig2.savefig('../figs/taylorf2ecc_amp.png')
    plt.show()
