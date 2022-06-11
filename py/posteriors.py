import numpy as np; import corner
import bajes; 

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True

def make_corner_plot(matrix, labels, range, color, limits=None, fig=None, bin=50):

    L = max(len(matrix[0]), len(np.transpose(matrix)[0]))
    N = int(min(len(matrix[0]), len(np.transpose(matrix)[0])))

    if fig == None:
        fig = cornerfig=corner.corner(matrix,
                                labels          = labels,
                                weights=np.ones(L)*100./L,
                                bins            = bin,
                                range           = range,
                                color           = color,
                                levels          = [.5, .9],
                                quantiles       = [.05, 0.5, .95],
                                contour_kwargs  = {'colors':color,'linewidths':0.95},
                                label_kwargs    = {'size':12.},
                                hist_kwargs     = {},
                                plot_datapoints = False,
                                show_titles     = True,
                                plot_density    = True,
                                smooth1d        = True,
                                smooth          = True)
    else:
        fig = cornerfig=corner.corner(matrix,
                                fig             = fig,
                                weights=np.ones(L)*100./L,
                                labels          = labels,
                                range           = range,
                                bins            = bin,
                                color           = color,
                                levels          = [.5, .9],
                                quantiles       = [.05, 0.5, .95],
                                contour_kwargs  = {'colors':color,'linewidths':0.95},
                                hist_kwargs     = {'color':'k'},
                                label_kwargs    = {'size':12.},
                                plot_datapoints = False,
                                show_titles     = True,
                                plot_density    = True,
                                smooth1d        = True,
                                smooth          = True)
    axes = np.array(cornerfig.axes).reshape((N,N))
    
    if(limits is not None):
        for i in np.arange(N):
            ax = axes[i, i]
            ax.set_ylim((0,limits[i]))

    return fig

latex_labels ={
        'm1'        : r'$m_1$  [$M_{\odot}$]',
        'm2'        : r'$m_2$  [$M_{\odot}$]',
        'mtot'      : r'$M$  [$M_{\odot}$]',
        'mchirp'    : r'$\mathcal{M}$ [$M_{\odot}$]',
        'q'         : r'$q$',
        'lambda1'   : r'$\Lambda_1$',
        'lambda2'   : r'$\Lambda_2$',
        'lambdat'   : r'$\tilde{\Lambda}$',
        'delta_lambda':r'$\delta\Lambda$',
        's1'        : r'$s_1$',
        's2'        : r'$s_2$',
        's1z'        : r'$s_1^z$',
        's2z'        : r'$s_2^z$',
        'chi1z'     : r'$\chi_1^z$',
        'chi1z'     : r'$\chi_1^z$',
        'chieff'    : r'$\chi_{\rm eff}$',
        'chip'      : r'$\chi_p$',
        'distance'  : r'$D_L$ [Mpc]',
        'energy'    : r'$\hat{E}^0$',
        'angmom'    : r'$p^0_{\varphi}$',
        'ecc'       : r'$e^0$',
        'omg0'      : r'$\omega^0$',
        'ra'        : r'ra',
        'dec'       : r'dec',
        'cosi'      : r'$\cos\iota$',
        'psi'       : r'$\psi$',
        'logL'      : r'$\log\mathcal{L}$',
        'logPrior'  : r'$\log\mathcal{P}$',
        'time_shift': r'$\Delta t$'
    }

class Posterior():
    """
    Class to plot and analyze posterior samples from
    Bajes
    """
    def __init__(self, path='./posterior.dat', opts=None) -> None:
        
        self.path = path
        self.opts = opts

        self.__read_posterior_file()
        
        # compute parameters 
        self.__compute_component_masses()
        if opts['aligned']:
            print("Compute aligned spins")
            self.__compute_align_spin_parameters()
        if opts['precessing']:
            print("Compute precessing spins")
            self.__compute_prec_spin_parameters()
        if opts['tidal']:
            print("Compute tides")
            self.__compute_tidal_parameters()
        
        pass

    def __read_posterior_file(self):
        """
        Read the posterior.dat in a dictionary
        """
        post   = np.genfromtxt(self.path, names=True)
        post_n = {key: post[key] for key in post.dtype.names}
        self.post = post_n

    def __compute_component_masses(self):
        m1 = bajes.obs.gw.utils.mcq_to_m1(self.post['mchirp'], self.post['q'])
        m2 = bajes.obs.gw.utils.mcq_to_m2(self.post['mchirp'], self.post['q'])
        self.post['m1'] = m1 
        self.post['m2'] = m2 

    def __compute_prec_spin_parameters(self):
        post  = self.post
        s1    = post['s1']
        s2    = post['s2']
        m1    = post['m1']
        m2    = post['m2']

        chi1z  = np.cos(post['tilt1'])*s1
        self.post['chi1z'] = chi1z 
        chi2z  = np.cos(post['tilt2'])*s2
        self.post['chi2z'] = chi2z
        chieff = bajes.obs.gw.utils.compute_chi_eff(m1, m2, chi1z, chi2z)
        self.post['chieff'] = chieff    
        chip   = [bajes.obs.gw.utils.compute_chi_prec(m1[i], m2[i], s1[i], s2[i], post['tilt1'][i], post['tilt2'][i]) for i in range(len(s1))]
        self.post['chip'] = chip

    def __compute_align_spin_parameters(self):
        post  = self.post
        chi1z = post['s1z']   
        chi2z = post['s2z']
        m1    = post['m1']
        m2    = post['m2']

        chieff = bajes.obs.gw.utils.compute_chi_eff(m1, m2, chi1z, chi2z)
        self.post['chieff'] = chieff

    def __compute_tidal_parameters(self):
        post  = self.post
        l1    = post['lambda1']
        l2    = post['lambda2']
        m1    = post['m1']
        m2    = post['m2']

        self.post['lambdat']        = bajes.obs.gw.utils.compute_lambda_tilde(m1, m2, l1, l2)
        self.post['delta_lambda']   = bajes.obs.gw.utils.compute_delta_lambda(m1, m2, l1, l2)

    def add_parameter(self, par_key, par_val, par_tex=None):
        self.post[par_key]    = par_val
        latex_labels[par_key] = par_tex

    def read_parameter(self, par_key):
        return self.post[par_key]
    
    def plot_hist(self):
        # TODO: plot a histogram of the parameter
        return None

    def plot_corner(self, parlist, color='b', figure=None):

        print(parlist)

        pars   = []
        labels = []   

        for i in parlist:
            if i in self.post.keys():
                pars.append(self.post[i])
                labels.append(latex_labels[i])
        
        return make_corner_plot(np.transpose(pars), labels, None, color, fig=figure)