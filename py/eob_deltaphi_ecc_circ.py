import matplotlib.pyplot as plt
import numpy as np
import eob_waveform as eobwf
from scipy import interpolate

def interp_eval(x, y, x0):
    f = interpolate.interp1d(x, y)
    return f(x0)

def run_once(M, q, lam1, lam2, initial_freq, omg_target, ecc):
    """
    Compute the phase difference at a reference BNS frequency
    between BNS and BBH system

    input:
        M            = total mass
        q            = mass ratio
        lam1, lam2   = tidal parameters
        initial_freq = initial frequency (geometric units)
        omg_target   = target GW frequency (Hz)
        ecc          = eccentricity (at *apastron*)
    
    returns:
        t_bns_tar    = dimensionless time at which BNS is at reference GW frequency
        phi_bns_tar  = phase of the BNS GW at t_bns_tar
        phi_bbh_tar  = phase of the BBH GW at t_bns_tar
        dp           = phi_bbh_tar - phi_bns_tar
        wf_bbh       = BBH waveform
        wf_bns       = BNS waveform
    """

    pars     =  eobwf.CreateDict(q=q, ecc=ecc, f0=initial_freq)
    wf_bbh   =  eobwf.Waveform_PyEOB(params=pars)
    pars     =  eobwf.CreateDict(q=q, ecc=ecc, f0=initial_freq, l1=lam1, l2=lam2)
    wf_bns   =  eobwf.Waveform_PyEOB(params=pars)

    _,phi_bns, omg_bns = wf_bns.load_hlm(2,2)
    omg_bns              = omg_bns/(M*eobwf.Msuns)

    # cut after max omg_bns_c
    imx      = np.argmax(omg_bns)
    omg_bns = omg_bns[:imx]
    phi_bns = phi_bns[:imx]

    # find f = omg_target for "M" Msun system 
    phi_bns_tar = interp_eval(omg_bns,  phi_bns, omg_target)
    t_bns_tar   = interp_eval(omg_bns,  wf_bns.t[:imx], omg_target)
    phi_bbh_tar = interp_eval(wf_bbh.t, wf_bbh.hlm['1'][1], t_bns_tar)
    dp          = phi_bbh_tar - phi_bns_tar
    #print("dp circc at " + str(omg_target)+ " Hz (BNS circ) = ", dp)
    return t_bns_tar, phi_bns_tar, phi_bbh_tar, dp, wf_bbh, wf_bns

if __name__ == "__main__":

    N    = 50
    M    = 2.7
    q    = 1.
    lam1 = 400
    lam2 = 400
    omg_target = 1500.
    initial_freq = 0.001

    phis   = np.zeros(N)
    phis_2 = np.zeros(N)
    omgs   = np.zeros(N)
    ttargs = np.zeros(N)
    es     = np.linspace(0.002, 0.01, N)

    j = 0

    #########################################
    # difference between circular waveforms
    #########################################
    t_bns_tar_c, _,_,dp_circ, wf_bbh_c, wf_bns_c = run_once(M, q, lam1, lam2, initial_freq, omg_target, 0.)
    print("dp circ at " + str(omg_target)+ " Hz (BNS circ) = ", dp_circ)

    f_bbh_c = interpolate.interp1d(wf_bbh_c.t, wf_bbh_c.hlm['1'][1])
    f_bns_c = interpolate.interp1d(wf_bns_c.t, wf_bns_c.hlm['1'][1])

    ###############################
    # repeat for eccentric
    ###############################

    for e_i in es:

        print(j, "Running ecc=", e_i)
        t_bns_tar_e,phi_bns_tar_e, phi_bbh_tar_e,dp,_,_ = run_once(M, q, lam1, lam2, initial_freq, omg_target, e_i)

        # phase of QC BBH and BNS at the time when eccentric BNS is at reference freq.
        phi_bbh_c=f_bbh_c(t_bns_tar_e)
        phi_bns_c=f_bns_c(t_bns_tar_e)

        phis[j]   = phi_bbh_tar_e - phi_bns_tar_e
        phis_2[j] = phi_bbh_c - phi_bns_c
        ttargs[j] = t_bns_tar_e
        print("\t...phi BBH - phi BNS =", phi_bbh_c-phi_bns_c)
        print("\t...Done")
        j = j+1


    # Save data to txt
    with open('ecc_dphi_001_900.txt','w') as f:
        for i in range(j):
            f.write('%f %f %f %f %f\n' %(es[i], phis[i],phis[i]-dp_circ, phis_2[i], phis[i] - phis_2[i]))
    f.close()

    # plot
    plt.plot(es, phis, '-o')
    plt.xlabel("eccentricity")
    plt.ylabel("$\Delta\Phi = \phi^{BBH}(t_f) - \phi^{BNS}(t_f)$")
    plt.show()

    plt.plot(es, phis-dp_circ, '-o')
    plt.xlabel("eccentricity")
    plt.ylabel("$\Delta\Phi - \Delta\Phi(e=0)$")
    plt.show()

    plt.plot(es, ttargs, '-o')
    plt.xlabel('eccentricity')
    plt.yabel('target time')
    plt.show()

    plt.plot(es, phis_2, '-o')
    plt.ylabel(r'$\Delta\Phi_c = \phi^{\rm BBH, circular}(t_f) - \phi^{\rm BNS, circular}(t_f)$')
    plt.xlabel("eccentricity")
    plt.show()

    plt.plot(es, phis - phis_2,  '-o')
    plt.ylabel(r"$\Delta\Phi - \Delta\Phi_c$")
    plt.xlabel("eccentricity")
    plt.show()
