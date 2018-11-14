import numpy as np
import matplotlib.pyplot as plt
import copy
import pdb
import astropy.stats
import radvel
import radvel.fitting

import utils
# import rvsearch.utils


class Periodogram:
    """Class to calculate and store periodograms.

    Args:
        posterior (radvel.Posterior): radvel.Posterior object
        minsearchp (float): minimum search period
        maxsearchp (float): maximum search period
        num_known_planets (int): Assume this many known planets in the system and search for one more
        num_pers (int): (optional) number of frequencies to test
            [default = calculated via rvsearch.periodograms.freq_spacing]
    """

    def __init__(self, post, basebic=None, num_known_planets=0, minsearchp=100, maxsearchp=10000,
                 baseline=True, basefactor=4., num_pers=None, search_pars=['per'],
                 valid_types = ['bic', 'aic', 'ls'], verbose=True):
        self.post = copy.deepcopy(post)
        self.default_pdict = {}
        for k in post.params.keys():
            self.default_pdict[k] = self.post.params[k].value

        self.basebic = basebic
        self.num_known_planets = num_known_planets

        self.times = self.post.likelihood.x
        self.vel = self.post.likelihood.y
        self.errvel = self.post.likelihood.yerr
        self.timelen = np.amax(self.times) - np.amin(self.times)

        self.tels = []
        for val in self.post.params.keys():
            if 'gamma_' in val:
                self.tels.append(val.split('_')[1])

        self.minsearchP = minsearchp
        self.maxsearchP = maxsearchp
        self.baseline = baseline
        self.basefactor = basefactor
        self.num_pers = num_pers

        if self.baseline == True:
            self.maxsearchP = self.basefactor * self.timelen

        self.search_pars = search_pars
        self.valid_types = valid_types
        self.power = {key: None for key in self.valid_types}

        self.verbose = verbose

        self.best_per = None
        self.best_bic = None

        self.bic_thresh = None

        # Automatically generate a period grid upon initialization.
        self.make_per_grid()

    @classmethod
    def from_pandas(cls, data):
        params = utils.initialize_default_pars(instnames=data.tel)
        post = utils.initialize_post(data, params=params)
        return cls(post)

    @classmethod
    def from_csv(cls, filename):
        data = utils.read_from_csv(filename)
        params = utils.initialize_default_pars(instnames=data.tel)
        post = utils.initialize_post(data, params=params)
        return cls(post)

    def per_spacing(self, oversampling=1, verbose=True):
        """Get the number of sampled frequencies and return a period grid

        Condition for spacing: delta nu such that during the
        entire duration of observations, phase slip is no more than P/4

        Args:
            oversampling (float): (optional) oversampling factor
            verbose (bool): (optional) print extra messages

        Returns:
            array: Array of test periods
        """

        fmin = 1. / self.maxsearchP
        fmax = 1. / self.minsearchP

        dnu       = 1. / (4. * self.timelen)
        num_freq  = int((fmax - fmin) / dnu + 1)
        num_freq *= oversampling
        num_freq  = int(num_freq)

        if verbose:
            print("Number of test periods:", num_freq)

        freqs = np.linspace(fmax, fmin, num_freq)
        pers = 1. / freqs

        self.num_pers = num_freq
        return pers

    def make_per_grid(self):
        if self.num_pers == None:
            self.pers = self.per_spacing()
        else:
            self.pers = 1/np.linspace(1/self.maxsearchP, 1/self.minsearchP, self.num_pers)
        self.freqs = 1/self.pers

    def per_bic(self):
        #BJ's method. Remove once final BIC/AIC method is established.
        """Compute delta-BIC periodogram. ADD: crit is BIC or AIC.
        """

        print("Calculating BIC periodogram")
        # This assumes nth planet parameters, and all periods, were locked in.
        # SET ALL PARS TO BE FIXED, EXCEPT gamma, jitter, dvdt, curv
        if self.basebic is None:
            self.post.params['per1'].vary = False
            self.post.params['tc1'].vary = False
            self.post.params['k1'].vary = False
            # Vary ONLY gamma, jitter, dvdt, curv. All else fixed, and k=0
            baseline_fit = radvel.fitting.maxlike_fitting(self.post, verbose=False)
            baseline_bic = baseline_fit.likelihood.bic()
        else:
            baseline_bic = self.basebic
        rms = np.std(self.post.likelihood.residuals())
        self.default_pdict['k{}'.format(self.post.params.num_planets)] = rms

        # Allow amplitude and time offset to vary, fix eccentricity and period.
        self.post.params['secosw{}'.format(self.num_known_planets+1)].vary = False
        self.post.params['sesinw{}'.format(self.num_known_planets+1)].vary = False

        self.post.params['k{}'.format(self.num_known_planets+1)].vary = True
        self.post.params['tc{}'.format(self.num_known_planets+1)].vary = True

        power = np.zeros_like(self.pers)
        ks = np.zeros_like(self.pers)
        tcs = np.zeros_like(self.pers)
        dvdts = np.zeros_like(self.pers)
        curvs = np.zeros_like(self.pers)
        jits = {tel:[] for tel in self.tels}
        gammas = {tel:[] for tel in self.tels}

        for i, per in enumerate(self.pers):
            if self.verbose:
                print(' {}'.format(i), '/', self.num_pers, end='\r')
            # Reset posterior parameters to default values.
            for k in self.default_pdict.keys():
                self.post.params[k].value = self.default_pdict[k]

            #Set new period, fix period, and fit a circular orbit.
            perkey = 'per{}'.format(self.num_known_planets+1)
            self.post.params[perkey].value = per
            self.post.params[perkey].vary = False

            fit = radvel.fitting.maxlike_fitting(self.post, verbose=False)
            power[i] = baseline_bic - fit.likelihood.bic()
            ks[i] = fit.params['k{}'.format(self.num_known_planets+1)].value
            tcs[i] = fit.params['tc{}'.format(self.num_known_planets+1)].value
            dvdts[i] = fit.params['dvdt'].value
            curvs[i] = fit.params['curv'].value
            for tel in self.tels:
                jits[tel].append(fit.params['k{}'.format(self.num_known_planets+1)].value)
                gammas[tel].append(fit.params['tc{}'.format(self.num_known_planets+1)].value)

        fit_index = np.argmax(power)
        self.best_per = self.pers[fit_index]
        self.best_k = ks[fit_index]
        self.best_tc = tcs[fit_index]
        self.best_dvdt = dvdts[fit_index]
        self.best_curv = curvs[fit_index]
        self.best_bic = power[fit_index]
        self.best_gamma = {tel:jits[tel][fit_index] for tel in self.tels}
        self.best_jit = {tel:gammas[tel][fit_index] for tel in self.tels}

        self.power['bic'] = power

    def ls(self):
        """Astropy Lomb-Scargle periodogram.
        """
        #FOR TESTING
        print("Calculating Lomb-Scargle periodogram")
        periodogram = astropy.stats.LombScargle(self.times, self.vel, self.errvel)
        power = periodogram.power(self.freq_array)
        #freqs = periodogram
        self.power['ls'] = power

    def eFAP_thresh(self, fap=0.01):
        """Calculate the threshold for significance based on BJ's eFAP algorithm
        From Lea's code. LOMB-S OPTION?
        """
        # select out intermediate values of BIC, median - 95%
        sBIC = np.sort(self.power['bic'])
        crop_BIC = sBIC[int(0.5*len(sBIC)):int(0.95*len(sBIC))]

        hist, edge = np.histogram(crop_BIC, bins=10)
        cent = (edge[1:]+edge[:-1])/2.
        norm = float(np.sum(hist))
        nhist = hist/norm
        loghist = np.log10(nhist)

        func = np.poly1d(np.polyfit(cent[np.isfinite(loghist)], loghist[np.isfinite(loghist)], 1))
        xmod = np.linspace(np.min(sBIC[np.isfinite(sBIC)]), 10.*np.max(sBIC), 10000)
        lfit = 10.**func(xmod)
        fap_min = 10.**func(sBIC[-1])*self.num_pers
        thresh = xmod[np.where(np.abs(lfit-fap/self.num_pers) == np.min(np.abs(lfit-fap/self.num_pers)))]
        self.bic_thresh = thresh

    def save_per(self, ls=False):
        if ls==False:
            try:
                # FIX THIS; SPECIFY DIRECTORY/NAME, NUMBER OF PLANETS IN FILENAME, AND ARRAY ORDERING
                np.savetxt((self.pers, self.power['bic']), filename='BIC_periodogram.csv')
            except:
                print('Have not generated a delta-BIC periodogram.')
        else:
            try:
                np.savetxt((self.pers, self.power['ls']), filename='LS_periodogram.csv')
            except:
                print('Have not generated a Lomb-Scargle periodogram.')

    def plot_per(self, ls=False, alias=True, save=False):
        # TO-DO: WORK IN AIC/BIC OPTION, INCLUDE IN PLOT TITLE
        peak = np.argmax(self.power['bic'])
        f_real = self.freqs[peak]

        fig, ax = plt.subplots()
        ax.plot(self.pers, self.power['bic'])
        ax.scatter(self.pers[peak], self.power['bic'][peak], label='{} days'.format(
                   np.round(self.pers[peak], decimals=1)))

        # If DBIC threshold has been calculated, plot.
        if self.bic_thresh is not None:
            ax.axhline(self.bic_thresh, ls=':', c='y', label=r'$\Delta$BIC threshold')
            upper = 1.05*max(np.amax(self.power['bic']), self.bic_thresh)
            ax.set_ylim([np.amin(self.power['bic']), upper])
        else:
            ax.set_ylim([np.amin(self.power['bic']), 1.05*np.amax(self.power['bic'])])
        ax.set_xlim([self.pers[0], self.pers[-1]])

        if alias:
            # Plot sidereal day, lunation period, and sidereal year aliases.
            colors = ['r', 'b', 'g']
            alias = [0.997, 29.531, 365.256]
            for i in np.arange(3):
                f_ap = f_real + 1./alias[i]
                f_am = f_real - 1./alias[i]
                ax.axvline(1./f_am, linestyle='--', c=colors[i], alpha=0.75,
                           label='{} day alias'.format(np.round(alias[i], decimals=1)))
                ax.axvline(1./f_ap, linestyle='--', c=colors[i], alpha=0.75)

        ax.legend(loc=0)
        ax.set_xscale('log')
        ax.set_xlabel('Period (days)')
        ax.set_ylabel(r'$\Delta$BIC')  # TO-DO: WORK IN AIC/BIC OPTION
        ax.set_title('Planet {} vs. planet {}'.format(self.num_known_planets+1, self.num_known_planets))

        # Store figure as object attribute, make separate saving functionality?
        self.fig = fig
        if save:
            # FINISH THIS, WRITE NAMING PROCEDURE
            fig.savefig('dbic{}.pdf'.format(self.num_known_planets+1))


# TO-DO: MOVE THIS INTO CLASS STRUCTURE, OR REMOVE IF UNNECESSARY
def setup_posterior(post, num_known_planets):
    """Setup radvel.posterior.Posterior object

    Prepare posterior object for periodogram calculations. Fix values for previously-known planets.

    Args:
        post (radvel.posterior.Posterior): RadVel posterior object. Can be initialized from setup file or loaded
            from a RadVel fit.
        num_known_planets (int): Number of previously known planets. Parameters for these planets will be fixed.

    Returns:
        tuple: (radvel.posterior object used as baseline fit, radvel.posterior used in search)
    """
    basis_pars = post.likelihood.params.basis.name.split()

    for i in range(1, post.params.num_planets + 1):
        for par in basis_pars:
            parname = "{}{}".format(par, i)
            post.params[parname].vary = False

            if par == 'k':
                post.params[parname].value = 0.0
            elif par == 'logk':
                post.params[parname].value = -9

    # return (base_post, search_post)
    return search_post
