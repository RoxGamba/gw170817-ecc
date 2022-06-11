"""
Generate plots with e.g.
python3 plot_posteriors.py --f ../PE/injections/R01/posterior.dat --mchirp --q --lambdat --eccentricity --tidal --aligned --color royalblue
"""

import numpy as np; import optparse as op; import matplotlib.pyplot as plt
import posteriors as pos

parser=op.OptionParser()
parser.add_option('--f', dest='fname', action='append', help='path to posteriors.dat')

# add options to parser

# intrinsic
parser.add_option('--M',        action='store_true', default=False, help='total mass')
parser.add_option('--q',        action='store_true', default=False, help='mass ratio')
parser.add_option('--mchirp',   action='store_true', default=False, help='chirp mass')
# spins
parser.add_option('--aligned',   action='store_true', default=False, help='aligned spins')
parser.add_option('--precessing',action='store_true', default=False, help='aligned spins')
parser.add_option('--s1',        action='store_true', default=False, help='spin magnitude of the primary')
parser.add_option('--s2',        action='store_true', default=False, help='spin magnitude of the secondary')
parser.add_option('--tilt1',     action='store_true', default=False, help='tilt1')
parser.add_option('--tilt2',     action='store_true', default=False, help='tilt2')
parser.add_option('--chi1z',     action='store_true', default=False, help='chi1z, prec')
parser.add_option('--chi2z',     action='store_true', default=False, help='chi2z, prec')
parser.add_option('--s1z',       action='store_true', default=False, help='chi1z, aligned')
parser.add_option('--s2z',       action='store_true', default=False, help='chi2z, aligned')
parser.add_option('--chip',      action='store_true', default=False, help='chi_prec')
parser.add_option('--chieff',    action='store_true', default=False, help='chi_eff')
# tides
parser.add_option('--tidal',   action='store_true', default=False,  help='tides on')
parser.add_option('--lambda1',  action='store_true', default=False, help='tidal parameter of the primary')
parser.add_option('--lambda2',  action='store_true', default=False, help='tidal parameter of the secondary')
parser.add_option('--lambdat',   action='store_true', default=False, help='lambdat')
parser.add_option('--delta_lambda',   action='store_true', default=False, help='delta_lambda')
# eccentricity
parser.add_option('--eccentricity', action='store_true', default=False, help='eccentricity at reference frequency')
parser.add_option('--omg0',     action='store_true', default=False, help='initial frequency of the approximant')
# hyp
parser.add_option('--energy',   action='store_true', default=False,help='Energy at r0')
parser.add_option('--angmom',   action='store_true', default=False,help='Angmom at r0')
# ext
parser.add_option('--distance', action='store_true', default=False, help='luminosity distance')
parser.add_option('--cosi',     action='store_true', default=False, help='cosine of inclination')
parser.add_option('--psi',      action='store_true', default=False, help='polarization')
parser.add_option('--ra',       action='store_true', default=False, help='right ascension')
parser.add_option('--dec',      action='store_true', default=False, help='declination')
# logL
parser.add_option('--logL',     action='store_true', default=False, help='log likelihood')
parser.add_option('--logPrior', action='store_true', default=False, help='log prior')
# plot all params
parser.add_option('--all',      dest='all_flag',  type='int', default=0, help='cornerplot for all parameters sampled')
# colors for plot
parser.add_option('--color',    dest='color', action='append', help='Color of the cornerplot')


(opts,args) = parser.parse_args()

# initialize no figure
f = None

# read the posterior.dat in a dictionary
for fl, cols in zip(opts.fname, opts.color):
    
    post = pos.Posterior(path=fl, opts=vars(opts))
    ks = []
    for i in vars(opts):
            if(opts.__getattribute__(i)):
                ks.append(i)
    f = post.plot_corner(ks, color=cols, figure=f)

plt.show()
