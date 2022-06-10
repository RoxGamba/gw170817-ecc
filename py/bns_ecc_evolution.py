import matplotlib.pyplot as plt
import eob_waveform as eobwf
Msuns = eobwf.Msuns

def mcq_to_m1(mc,q):
    return mc*(q*q*(1.+q))**0.2
def mcq_to_m2(mc,q):
    return mc*((1.+q)/(q**3.))**0.2

# Parameters
mc   = 1.1977
q    = 1.5
lam1 = 400
lam2 = 600
f0   = 23.

m1 = mcq_to_m1(mc, q)
m2 = mcq_to_m2(mc, q)
M  = m1+m2

initial_freq = f0*M*Msuns

for ecc in [0.1, 0.05, 0.01]:
    print("Running ", ecc)
    pars      =  eobwf.CreateDict(q=q, ecc=ecc, f0=initial_freq, l1=lam1, l2=lam2)
    h_eob     =  eobwf.Waveform_PyEOB(params=pars)
    _,_,omg22 = h_eob.load_hlm(2,2)

    # Compute the eccentricities
    ecc_radial            = h_eob.ecc_radial()
    e_pn                  = h_eob.ecc_3PN()

    # Plot
    p= plt.plot(h_eob.t, ecc_radial,    label=r'$e=$ '+str(ecc))
    plt.plot(h_eob.dyn['t'],  e_pn,  linestyle='--', color=p[0].get_color())

plt.xlabel(r'$t/M $')
plt.ylabel('e')
plt.legend()
plt.savefig('../figs/bns_ecc_evolution.png')

