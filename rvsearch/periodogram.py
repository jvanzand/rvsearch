import copy
import pdb

import numpy as np
import matplotlib.pyplot as plt
import astropy.stats
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import radvel
import radvel.fitting
from radvel.plot import orbit_plots
from tqdm import tqdm
import pathos.multiprocessing as mp
from multiprocessing import Value
from itertools import repeat
from functools import partial

import rvsearch.utils as utils


class TqdmUpTo(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)`."""
    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize


class Periodogram(object):
    """Class to calculate and store periodograms.

    Args:
        post (radvel.Posterior): radvel.Posterior object
        minsearchp (float): minimum search period
        maxsearchp (float): maximum search period
        baseline (bool): Whether to calculate maxsearchp from obs. baseline
        basefactor (float): How far past the obs. baseline to search
        oversampling (float): By how much to oversample the period grid
        manual_grid (array): Option to feed in a user-chosen period grid
        fap (float): False-alarm-probability threshold for detection
        num_pers (int): (optional) number of frequencies to test, default
        eccentric (bool): Whether to fit with free or fixed eccentricity
        workers (int): Number of cpus over which to parallelize
        verbose (bool): Whether to print progress during calculation

    """

    def __init__(self, post, basebic=None, minsearchp=3, maxsearchp=10000,
                 baseline=True, basefactor=5., oversampling=1., manual_grid=None,
                 fap=0.001, num_pers=None, eccentric=False, workers=1,
                 verbose=True, starname='star'):
        self.starname = starname
        self.post = copy.deepcopy(post)
        self.default_pdict = {}
        for k in post.params.keys():
            self.default_pdict[k] = self.post.params[k].value

        self.basebic = basebic
        self.num_known_planets = self.post.params.num_planets - 1

        self.times = self.post.likelihood.x
        self.vel = self.post.likelihood.y
        self.errvel = self.post.likelihood.yerr
        self.timelen = np.amax(self.times) - np.amin(self.times)

        self.tels = np.unique(self.post.likelihood.telvec)
        '''
        for val in self.post.params.keys():
            if 'gamma_' in val:
                self.tels.append(val.split('_')[1])
        '''

        self.minsearchP = minsearchp
        self.maxsearchP = maxsearchp
        self.baseline = baseline
        self.basefactor = basefactor
        self.oversampling = oversampling
        self.manual_grid = manual_grid
        self.fap = fap
        self.num_pers = num_pers
        if self.manual_grid is not None:
            self.num_pers = len(manual_grid)

        self.eccentric = eccentric

        if self.baseline == True:
            self.maxsearchP = self.basefactor * self.timelen

        self.valid_types = ['bic', 'aic', 'ls']
        self.power = {key: None for key in self.valid_types}

        self.workers = workers
        self.verbose = verbose

        self.best_per = None
        self.best_bic = None

        self.bic_thresh = None
        # Pre-compute good-fit floor of the BIC periodogram.
        if self.eccentric:
            self.floor = -8*np.log(len(self.times))
        else:
            self.floor = -4*np.log(len(self.times))

        # Automatically generate a period grid upon initialization.
        self.make_per_grid()

    def per_spacing(self):
        """Get the number of sampled frequencies and return a period grid.

        Condition for spacing: delta nu such that during the
        entire duration of observations, phase slip is no more than P/4

        Returns:
            array: Array of test periods

        """
        fmin = 1./self.maxsearchP
        fmax = 1./self.minsearchP

        # Should be 1/(2*pi*baseline), was previously 1/4.
        dnu       = 1./(2*np.pi*self.timelen)
        num_freq  = (fmax - fmin)/dnu + 1
        num_freq *= self.oversampling
        num_freq  = int(num_freq)

        if self.verbose:
            print("Number of test periods:", num_freq)

        freqs = np.linspace(fmax, fmin, num_freq)
        pers = 1. / freqs

        self.num_pers = num_freq
        return pers

    def make_per_grid(self):
        """Generate a grid of periods for which to compute likelihoods.

        """
        if self.manual_grid is not None:
            self.pers = np.array(self.manual_grid)
        else:
            if self.num_pers is None:
                self.pers = self.per_spacing()

            else:
                self.pers = (1/np.linspace(1/self.maxsearchP, 1/self.minsearchP,self.num_pers))[::-1]
                

        self.freqs = 1/self.pers

    def per_bic(self):
        """Compute delta-BIC periodogram. ADD: crit is BIC or AIC.

        """

        prvstr = str(self.post.params.num_planets-1)
        plstr = str(self.post.params.num_planets)
        
        #if "gamma_k" not in self.post.params.keys():
            #print("Entered per_bic", self.post.params.keys())
            #fsdfds
        if self.verbose:
            print("Calculating BIC periodogram for {} planets vs. {} planets".format(plstr, prvstr))
        # This assumes nth planet parameters, and all periods, are fixed.
        if self.basebic is None:
            # Handle the case where there are no known planets.
            if self.post.params.num_planets == 1 and self.post.params['k1'].value == 0.:
                self.post.params['per'+plstr].vary = False
                self.post.params['tc'+plstr].vary = False
                self.post.params['k'+plstr].vary = False
                self.post.params['secosw'+plstr].vary = False
                self.post.params['sesinw'+plstr].vary = False
                # Vary ONLY gamma, jitter, dvdt, curv. All else fixed, and k=0
                baseline_fit = radvel.fitting.maxlike_fitting(self.post, verbose=False)
                baseline_bic = baseline_fit.likelihood.bic()

            # Handle the case where there is at least one known planet.
            else:
                self.post.params['per{}'.format(self.num_known_planets+1)].vary = False
                self.post.params['tc{}'.format(self.num_known_planets+1)].vary = False
                self.post.params['k{}'.format(self.num_known_planets+1)].vary = False
                self.post.params['secosw{}'.format(self.num_known_planets+1)].vary = False
                self.post.params['sesinw{}'.format(self.num_known_planets+1)].vary = False
                baseline_bic = self.post.likelihood.bic()

        else:
            baseline_bic = self.basebic

        rms = np.std(self.post.likelihood.residuals())
        self.default_pdict['k{}'.format(self.post.params.num_planets)] = rms

        #print("The baseline!", baseline_bic)
        #import pdb; pdb.set_trace()


        # Allow amplitude and time offset to vary, fix period (and ecc. if asked.)
        self.post.params['per{}'.format(self.num_known_planets+1)].vary = False
        if self.eccentric == True:
            # If eccentric set to True, Free eccentricity.
            self.post.params['secosw{}'.format(self.num_known_planets+1)].vary = True
            self.post.params['sesinw{}'.format(self.num_known_planets+1)].vary = True
        else:
            # If eccentric set to False, fix eccentricity to zero.
            self.post.params['secosw{}'.format(self.num_known_planets+1)].vary = False
            self.post.params['sesinw{}'.format(self.num_known_planets+1)].vary = False

        self.post.params['k{}'.format(self.num_known_planets+1)].vary  = True
        self.post.params['tc{}'.format(self.num_known_planets+1)].vary = True

        # Divide period grid into as many subgrids as there are parallel workers.
        self.sub_pers = np.array_split(self.pers, self.workers)
        #print("YEEEER", self.pers)
        #sdfd
        #import pdb; pdb.set_trace()


        single_use = False
        if single_use:
            #### For single use ######
            post = copy.deepcopy(self.post)
            #per_array = self.sub_pers[n]
            #fit_params = [{} for x in range(len(per_array))]
            #bic = np.zeros_like(per_array)

            for p in post.params.keys():
                if "gamma" in p:
                    post.params[p].vary = True

            per = 444
            for k in self.default_pdict.keys():
                post.params[k].value = self.default_pdict[k]
                
            perkey = 'per{}'.format(self.num_known_planets+1)
            post.params[perkey].value = per

            post.params['jit_j'].vary = True

            print("Pre-Paramssss", post.params['per2'].value, post.params['k2'].value, post.params['k2'].vary)
            ##
            data_x = post.likelihood.x
            data_y = post.likelihood.y
            #print("Check 2: ", data_x[11], data_y[11])
            model_x = np.linspace(np.min(data_x), np.max(data_x), 600)
            model_y = post.likelihood.model(model_x) + post.params['gamma_j'].value
            #model_y = post.likelihood.model(post.likelihood.x) + post.params['gamma_j'].value
            import matplotlib.pyplot as plt
            plt.close()
                   
            plt.scatter(data_x, data_y, c='r', label='data')
            plt.plot(model_x, model_y, c='b', label='model')
            plt.legend()
            plt.savefig("data_model_compare_pre.png")
            plt.close()

            post = radvel.fitting.maxlike_fitting(post, verbose=False)


            plt.close()
            data_x = post.likelihood.x
            data_y = post.likelihood.y
            model_x = np.linspace(np.min(data_x), np.max(data_x)+600, 2000)
            model_y = post.likelihood.model(model_x) + post.params['gamma_j'].value
            #import pdb; pdb.set_trace()

            plt.scatter(data_x, data_y, c='r', label='data')
            plt.plot(model_x, model_y, c='b', label='model')
            plt.legend()
            plt.savefig("data_model_compare_post.png")
            plt.close()
            ##########################
            bic = baseline_bic - post.likelihood.bic()
            #print("THE single BIC", baseline_bic, post.likelihood.bic, bic)
            #import pdb; pdb.set_trace()
            #sdfsd
        
        # Define a function to compute periodogram for a given grid section.
        def _fit_period(n):
            post = copy.deepcopy(self.post)
            per_array = self.sub_pers[n]
            fit_params = [{} for x in range(len(per_array))]
            bic = np.zeros_like(per_array)

            

            ## Judah addition: let offsets float to avoid analytic value in radvel.likelihood.residuals
            for p in post.params.keys():
                if "gamma" in p:
                    post.params[p].vary = True

            
            for i, per in enumerate(per_array):

                
                # Reset posterior parameters to default values.
                for k in self.default_pdict.keys():
                    post.params[k].value = self.default_pdict[k]

                    
                    #if "jit" in k:
                       # if post.params[k].value>10:
                           # post.params[k].value = 5

                     
                
                perkey = 'per{}'.format(self.num_known_planets+1)
                post.params[perkey].value = per
 

                plott=False
                if plott:
                    if 107<per<108:
                        print("Pre-Paramssss", post.params['k1'].value, post.params['k1'].vary, post.params['k2'].value, post.params['k2'].vary)


                        #post.params['jit_j'].value=5
                        #post.params['jit_j'].vary=True


                        ##
                        data_x = post.likelihood.x
                        data_y = post.likelihood.y
                        #print("Check 2: ", data_x[11], data_y[11])
                        model_x = np.linspace(np.min(data_x), np.max(data_x), 2000)
                        model_y = post.likelihood.model(model_x) + post.params['gamma_j'].value
                        #model_y = post.likelihood.model(post.likelihood.x) + post.params['gamma_j'].value
                        import matplotlib.pyplot as plt
                        plt.close()
                   
                        plt.plot(model_x, model_y, c='b', label='model', zorder=1)
                        plt.scatter(data_x, data_y, c='r', label='data', zorder=2)

                        plt.legend()
                        plt.savefig("data_model_compare_pre_{}.png".format(per))
                        plt.close()

                        #print("AND MY gamma pre", post.params['gamma_j'].value, post.params['gamma_apf'].value)

                post = radvel.fitting.maxlike_fitting(post, verbose=False)

                #print(self.pers)
                if plott:
                    if 107<per<108:
                        #print("Post-params", post.params['k2'], post.params['dvdt'], post.params['curv'])
                        print("Post-Paramssss", post.params['k1'].value, post.params['k1'].vary, post.params['k2'].value, post.params['k2'].vary)
                        #print("Post-Paramssss", post.params['per1'].value, post.params['k1'].value, post.params['k1'].vary)
                        #print('')

                        #print("model BIC", post.likelihood.bic(), baseline_bic)

                        plt.close()
                        data_x = post.likelihood.x
                        data_y = post.likelihood.y
                        model_x = np.linspace(np.min(data_x), np.max(data_x), 2000)
                        model_y = post.likelihood.model(model_x) + post.params['gamma_j'].value
                        #import pdb; pdb.set_trace()

                        plt.plot(model_x, model_y, c='b', label='model', zorder=1)
                        plt.scatter(data_x, data_y, c='r', label='data', zorder=2)
                        print("AND MY Num pleez", post.params.num_planets)

                        plt.legend()
                        plt.savefig("data_model_compare_post_{}.png".format(per))
                        plt.close()
                        print("THE single BIC", baseline_bic, post.likelihood.bic(), baseline_bic-post.likelihood.bic())
                        #sdfsdfs

                    
                bic[i] = baseline_bic - post.likelihood.bic()
                #print(baseline_bic, post.likelihood.bic(), bic[i])

                if bic[i] < self.floor - 1:

                    # If the fit is bad, reset k_n+1 = 0 and try again.
                    for k in self.default_pdict.keys():
                        post.params[k].value = self.default_pdict[k]
                    post.params[perkey].value = per

                    post.params['k{}'.format(post.params.num_planets)].value = 0
                    post = radvel.fitting.maxlike_fitting(post, verbose=False)
                    bic[i] = baseline_bic - post.likelihood.bic()
                        
                    
                    
                if bic[i] < self.floor - 1:

                    # If the fit is still bad, reset tc to better value and try again.
                    for k in self.default_pdict.keys():
                        post.params[k].value = self.default_pdict[k]

                        if "gamma" in k:
                            post.params[k].vary = False

                    post.params[perkey].value = per # Judah add: change period from its default_pdict value. Otherwise, it always gets set to 100 days.
                     
                    post.params['k{}'.format(post.params.num_planets)].value = rms # Judah delete
                    post.params['k{}'.format(post.params.num_planets)].vary = False                
                    veldiff = np.absolute(post.likelihood.y - np.median(post.likelihood.y))
                    tc_new = self.times[np.argmin(veldiff)]
                    post.params['tc{}'.format(post.params.num_planets)].value = tc_new
                    post = radvel.fitting.maxlike_fitting(post, verbose=False)
                    bic[i] = baseline_bic - post.likelihood.bic()


                #print("BICCCC", bic1,  bic2, bic3, per) # See if all BICS are coming out the same

                # Append the best-fit parameters to the period-iterated list.
                best_params = {}
                for k in post.params.keys():
                    best_params[k] = post.params[k].value
                fit_params[i] = best_params

                if self.verbose:
                    counter.value += 1
                    pbar.update_to(counter.value)

            return (bic, fit_params)
        
        if self.verbose:
            global pbar
            global counter

            counter = Value('i', 0, lock=True)
            pbar = TqdmUpTo(total=len(self.pers), position=0)

        if self.workers == 1:
            #print("In periodogram", self.post.params.keys())
            #sfsdfsd
            # Call the periodogram loop on one core.
            self.bic, self.fit_params = _fit_period(0)

        else:
            # Parallelize the loop over sections of the period grid.
            p = mp.Pool(processes=self.workers)
            output = p.map(_fit_period, (np.arange(self.workers)))

            # Sort output.
            all_bics = []
            all_params = []
            for chunk in output:
                all_bics.append(chunk[0])
                all_params.append(chunk[1])
            self.bic = [y for x in all_bics for y in x]
            self.fit_params = [y for x in all_params for y in x]

            # Close the pool object.
            p.close()

        #import pdb; pdb.set_trace()
        fit_index = np.argmax(self.bic)
        self.bestfit_params = self.fit_params[fit_index]
        self.best_bic = self.bic[fit_index]
        self.power['bic'] = self.bic

        self.ls()
        ls_best_per_ind = np.argmax(np.flip(self.power['ls']))
        self.ls_best_per = self.pers[ls_best_per_ind]
        self.bic_best_per = self.pers[fit_index]


        if self.verbose:
            pbar.close()


    def ls(self):
        """Compute Lomb-Scargle periodogram with astropy.

        """
        # FOR TESTING
        #print("Calculating Lomb-Scargle periodogram")
        periodogram = astropy.stats.LombScargle(self.times, self.vel,
                                                self.errvel)
        power = periodogram.power(np.flip(self.freqs))
        self.power['ls'] = power

    def eFAP(self):
        """Calculate the threshold for significance based on BJ's empirical
            false-alarm-probability algorithm, and estimate the
            false-alarm-probability of the DBIC global maximum.

            Modified version by JB Ruffio (2022-02-17) based on the integral of an exponential decay.

        """

        sBIC = np.sort(self.power['bic'])
        crop_BIC = sBIC[int(0.5 * len(sBIC)):int(0.95 * len(sBIC))]
        med_BIC = crop_BIC[0]
        #print(len(sBIC), np.min(sBIC), np.max(sBIC), len(crop_BIC))
        #import pdb; pdb.set_trace()

        hist, edge = np.histogram(crop_BIC-med_BIC, bins=10)
        cent = (edge[1:] + edge[:-1]) / 2.

        loghist = np.log10(hist)
        a,b = np.polyfit(cent[np.isfinite(loghist)], loghist[np.isfinite(loghist)], 1)
        #import pdb; pdb.set_trace()

        if a>0: # Judah: If the line is positively sloped, use the lower sBIC values and try again
          #print("Found positive slopeeeeeeee")
          crop_BIC = sBIC[int(0 * len(sBIC)):int(0.95 * len(sBIC))]
          med_BIC = crop_BIC[0]
          hist, edge = np.histogram(crop_BIC-med_BIC, bins=10)
          cent = (edge[1:] + edge[:-1]) / 2.

          if hist[-1]>hist[-2]:
              hist = hist[:-1]
              edge = edge[:-1]
              cent = (edge[1:] + edge[:-1]) / 2.

          loghist = np.log10(hist)
          a,b = np.polyfit(cent[np.isfinite(loghist)], loghist[np.isfinite(loghist)], 1)
          
          xlab = "crop_BIC - min_BIC"

        else:
          xlab = "crop_BIC - med_BIC"


        ## Another of Judahs ideas
        # Leaving for now: sometimes interpolation doesn't go down to 0.001, causing error. Try 100x (crop_BIC-med_BIC)
        #from scipy.interpolate import interp1d

        #extrap_x = np.linspace((crop_BIC-med_BIC)[0], 10*(crop_BIC-med_BIC)[-1], 1000)
        #extrap_y = np.e**(a*extrap_x+b) / len(crop_BIC)

        #bic_from_fap = interp1d(extrap_y, extrap_x)
        #judah_bic = bic_from_fap(0.001)+med_BIC

        #self.bic_thresh = judah_bic

        

    
        
        
        B=10**b
        A=-a*np.log(10)
        #print("Boutta thresh", self.fap, self.num_pers, A, med_BIC)
        self.bic_thresh = np.log(self.fap / self.num_pers) / (-A)+med_BIC
        self.fap_min = np.exp(-A*(sBIC[-1]-med_BIC)) * self.num_pers
        #print("BIC THresh and FAP", self.bic_thresh, self.fap_min)

        #import pdb; pdb.set_trace()
        #print("bic_thresh and judah bic: ", self.bic_thresh, judah_bic)
        
        # If a is still >0 after the catch above, then set manually. Put this statement after self.fap_min calculation so it gets calculated normally.
        if a>0:
          print("Uh oh, slope still positive. Setting self.bic_thresh=30")
          self.bic_thresh = 30

        # Judah: extra check on huge BIC thresholds. For T001694, most BIC values are ~-1e8 because they don't. This causes the BIC threshold to be +7e8.

        #print("Thresh compare", self.bic_thresh)
        #sdfsdfd
        plott=False
        if plott:
            plt.close()
            #hist, edge = np.histogram(crop_BIC-med_BIC, bins=10)
            #cent = (edge[1:] + edge[:-1]) / 2.
            #loghist = np.log10(hist)
            w = (np.max(crop_BIC-med_BIC)-np.min(crop_BIC-med_BIC))/20

            plt.bar(cent[np.isfinite(loghist)], loghist[np.isfinite(loghist)], width=w)


            bic_grid = np.linspace(np.min(sBIC), 1*np.max(sBIC), 200) - med_BIC
            plt.plot(bic_grid, a*bic_grid+b, c='red')
            plt.xlabel(xlab)
            plt.ylabel("log(N)")
            plt.title('ΔBIC_threshold={:.2f}'.format(self.bic_thresh))
            plt.savefig(self.starname+"_hist.png")
            plt.close()

            plt.plot(self.pers, self.power['bic'])
            plt.xlabel("period (days)")
            plt.ylabel("ΔBIC")
            plt.title('ΔBIC_threshold={:.2f}'.format(self.bic_thresh))
            
            plt.savefig(self.starname+"_bic_periodogram.png")
            #sdfsd
            #import pdb; pdb.set_trace()


    def save_per(self, filename, ls=False):
        df = pd.DataFrame([])
        df['period'] = self.pers
        if not ls:
            try:
                np.savetxt((self.pers, self.power['bic']), filename=\
                                                'BIC_periodogram.csv')
            except:
                print('Have not generated a delta-BIC periodogram.')
        else:
            try:
                df['power'] = self.power['ls']
            except KeyError:
                print('Have not generated a Lomb-Scargle periodogram.')

    def plot_per(self, alias=True, floor=True, save=False):
        """Plot periodogram.

        Args:
            alias (bool): Plot year, month, day aliases?
            floor (bool): Set y-axis minimum according to likelihood limit?
            save (bool): Save plot to current directory?

        """
        # TO-DO: WORK IN AIC/BIC OPTION, INCLUDE IN PLOT TITLE
        peak = np.argmax(self.power['bic'])
        f_real = self.freqs[peak]

        fig, ax = plt.subplots()
        ax.plot(self.pers, self.power['bic'])
        ax.scatter(self.pers[peak], self.power['bic'][peak], label='{} days'\
                            .format(np.round(self.pers[peak], decimals=1)))

        # If DBIC threshold has been calculated, plot.
        if self.bic_thresh is not None:
            ax.axhline(self.bic_thresh, ls=':', c='y', label='{} FAP'\
                                                    .format(self.fap))
            upper = 1.1*max(np.amax(self.power['bic']), self.bic_thresh)
        else:
            upper = 1.1*np.amax(self.power['bic'])

        if floor:
            # Set periodogram plot floor according to circular-fit BIC min.
            # Set this until we figure out how to fix known planet offset. 5/8
            lower = max(self.floor, np.amin(self.power['bic']))
        else:
            lower = np.amin(self.power['bic'])

        ax.set_ylim([lower, upper])
        ax.set_xlim([self.pers[0], self.pers[-1]])

        if alias:
            # Plot sidereal day, lunation period, and sidereal year aliases.
            colors = ['r', 'b', 'g']
            alias = [0.997, 29.531, 365.256]
            if np.amin(self.pers) <= 1.:
                alii = np.arange(1, 3)
            else:
                alii = np.arange(3)
            for i in alii:
                f_ap = 1./alias[i] + f_real
                f_am = 1./alias[i] - f_real
                ax.axvline(1./f_am, linestyle='--', c=colors[i], alpha=0.5,
                                label='{} day alias'.format(np.round(alias[i],
                                decimals=1)))
                ax.axvline(1./f_ap, linestyle='--', c=colors[i], alpha=0.5)

        ax.legend(loc=0)
        ax.set_xscale('log')
        ax.set_xlabel('Period (days)')
        ax.set_ylabel(r'$\Delta$BIC')  # TO-DO: WORK IN AIC/BIC OPTION
        ax.set_title('Planet {} vs. planet {}'.format(self.num_known_planets+1,
                                                      self.num_known_planets))

        formatter = ticker.ScalarFormatter()
        formatter.set_scientific(False)
        ax.xaxis.set_major_formatter(formatter)

        # Store figure as object attribute, make separate saving functionality?
        self.fig = fig
        if save:
            fig.savefig('dbic{}.pdf'.format(self.num_known_planets+1))
