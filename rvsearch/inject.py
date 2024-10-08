"""Injection and recovery class"""

import os
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
import pickle
import pathos.multiprocessing as mp
from multiprocessing import Value
import radvel
from .periodogram import TqdmUpTo

import rvsearch.utils
from rvsearch import search


class Injections(object):
    """
    Class to perform and record injection and recovery tests for a planetary system.

    Args:
        searchpath (string): Path to a saved rvsearch.Search object
        plim (tuple): lower and upper period bounds for injections
        klim (tuple): lower and upper k bounds for injections
        elim (tuple): lower and upper e bounds for injections
        num_sim (int): number of planets to simulate
        verbose (bool): show progress bar
    """

    def __init__(self, searchpath, plim, klim, elim, num_sim=1, full_grid=True, verbose=True, beta_e=False):
        self.searchpath = searchpath
        self.plim = plim
        self.klim = klim
        self.elim = elim
        self.num_sim = num_sim
        self.full_grid = full_grid
        self.verbose = verbose
        self.beta_e = beta_e

        #sub_trend = True

        ## Judah idea to inject into residuals only.
        use_resid=False
        if use_resid:
            original_search = pickle.load(open(searchpath, 'rb'))
            data = original_search.data
            data['mnvel'] = original_search.post.likelihood.residuals()
            starname = original_search.starname
            pmin, pmax = self.plim
            mstar = original_search.mstar
            mstar_err = original_search.mstar_err

            if False:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots()
                radvel.mtelplot(data.time, data.mnvel, data.errvel, data.tel, ax)
                plt.show()
                sfdf
            
            #import pdb; pdb.set_trace()
            resid_search = search.Search(data, min_per=pmin, max_per=pmax, baseline=False,
                                         sub_trend=False, trend=True, max_planets=8,
                                         workers=125, mcmc=False, verbose=False, mstar=[mstar, mstar_err],
                                         save_plots=False, eccentric=True)

            resid_search.pers = original_search.pers
            #import pdb; pdb.set_trace()

            self.search = resid_search # Update self.search



        else:
            self.search = pickle.load(open(searchpath, 'rb'))



        #self.search.polish = False # Judah experiment for speed.
        # Judah: fix previously discovered planet params
        #import pdb; pdb.set_trace()
        self.search.params_init   = self.search.post.params
        self.search.priors        = self.search.post.priors
        self.search.setup_planets = self.search.post.params.num_planets
        self.search.trend_and_planet_allowed = False ## For injections, disallow both trend and curve
        self.search.trend_subtractor()



        self.search.verbose = False
        seed = np.round(self.search.data['time'].values[0] * 1000).astype(int)

        self.injected_planets = self.random_planets(seed)
        self.recoveries = self.injected_planets

        self.outdir = os.path.dirname(searchpath)

        ## Save resid search so it can be loaded in run_injections()
        self.searchpath = searchpath.replace("search.pkl", "resid_search.pkl")
        pickle_out = open(self.searchpath,'wb')
        pickle.dump(self.search, pickle_out)
        pickle_out.close()


    def random_planets(self, seed):
        """Generate random planets

        Produce a DataFrame with random planet parameters

        Args:
            seed (int): seed for random number generator

        Returns:
            DataFrame: with columns inj_period, inj_tp, inj_e, inj_w, inj_k

        """

        p1, p2 = self.plim
        k1, k2 = self.klim
        e1, e2 = self.elim
        num_sim = self.num_sim
        beta_e = self.beta_e

        np.random.seed(seed)

        if p1 == p2:
            sim_p = np.zeros(num_sim) + p1
        else:
            sim_p = 10 ** np.random.uniform(np.log10(p1), np.log10(p2), size=num_sim)

        if k1 == k2:
            sim_k = np.zeros(num_sim) + k1
        else:
            sim_k = 10 ** np.random.uniform(np.log10(k1), np.log10(k2), size=num_sim)

        if beta_e:
            a = 0.867
            b = 3.03
            sim_e = np.random.beta(a, b, size=num_sim)
        else:
            if e1 == e2:
                sim_e = np.zeros(num_sim) + e1
            else:
                sim_e = np.random.uniform(e1, e2, size=num_sim)

        sim_tp = np.random.uniform(0, sim_p, size=num_sim)
        sim_om = np.random.uniform(0, 2 * np.pi, size=num_sim)

        df = pd.DataFrame(dict(inj_period=sim_p, inj_tp=sim_tp, inj_e=sim_e,
                               inj_w=sim_om, inj_k=sim_k))

        return df

    def run_injections(self, num_cpus=1):
        """Launch injection/recovery tests

        Try to recover all planets defined in self.simulated_planets

        Args:
            num_cpus (int): number of CPUs to utilize. Each injection will run
                on a separate CPU. Individual injections are forced to be single-threaded
        Returns:
            DataFrame: summary of injection/recovery tests

        """

        #import pdb; pdb.set_trace()

        def _run_one(orbel):
            sfile = open(self.searchpath, 'rb')
            search = pickle.load(sfile)
            search.verbose = False
            sfile.close()
            #search = self.search
            #print("Plannssss", self.search.num_planets)
            #sdfds
            recovered, recovered_orbel, trend_pref, trendel = search.inject_recover(orbel, num_cpus=1, full_grid=self.full_grid)

            last_bic = max(search.best_bics.keys())
            bic = search.best_bics[last_bic]
            thresh = search.bic_threshes[last_bic]

            if self.verbose:
                counter.value += 1
                pbar.update_to(counter.value)

            return recovered, recovered_orbel, bic, thresh, trend_pref, trendel

        outcols = ['inj_period', 'inj_tp', 'inj_e', 'inj_w', 'inj_k',
                   'rec_period', 'rec_tp', 'rec_e', 'rec_w', 'rec_k',
                   'recovered', 'bic']
        outdf = pd.DataFrame([], index=range(self.num_sim),
                             columns=outcols)
        outdf[self.injected_planets.columns] = self.injected_planets
        
        in_orbels = []
        out_orbels = []
        recs = []
        bics = []
        threshes = []
        trend_prefs = []
        trendels = []

        for i, row in self.injected_planets.iterrows():
            in_orbels.append(list(row.values))

        if self.verbose:
            global pbar
            global counter

            counter = Value('i', 0, lock=True)
            pbar = TqdmUpTo(total=len(in_orbels), position=0)

        pool = mp.Pool(processes=num_cpus)
        #import pdb; pdb.set_trace()
        outputs = pool.map(_run_one, in_orbels)

        for out in outputs:
            recovered, recovered_orbel, bic, thresh, trend_pref, trendel = out
            out_orbels.append(recovered_orbel)
            recs.append(recovered)
            bics.append(bic)
            threshes.append(thresh)
            trend_prefs.append(trend_pref)
            trendels.append(trendel)

        out_orbels = np.array(out_orbels)
        outdf['rec_period'] = out_orbels[:, 0]
        outdf['rec_tp'] = out_orbels[:, 1]
        outdf['rec_e'] = out_orbels[:, 2]
        outdf['rec_w'] = out_orbels[:, 3]
        outdf['rec_k'] = out_orbels[:, 4]

        outdf['recovered'] = recs
        outdf['bic'] = bics
        outdf['bic_thresh'] = threshes
        
        trendels = np.array(trendels)
        outdf['dvdt'] = trendels[:, 0]
        outdf['curv'] = trendels[:, 1]
        
        outdf['trend_pref'] = trend_prefs

        self.recoveries = outdf

        return outdf

    def save(self):
        self.recoveries.to_csv(os.path.join('recoveries.csv'), index=False)


class Completeness(object):
    """Calculate completeness surface from a suite of injections

    Args:
        recoveries (DataFrame): DataFrame with injection/recovery tests from Injections.save
    """

    def __init__(self, recoveries, xcol='inj_au', ycol='inj_msini',
                 mstar=None, rstar=None, teff=None, searches=None):#, trends_count=False):
        """Object to handle a suite of injection/recovery tests

        Args:
            recoveries (DataFrame): DataFrame of injection/recovery tests from Injections class
            mstar (float): (optional) stellar mass to use in conversion from p, k to au, msini
            xcol (string): (optional) column name for independent variable. Completeness grids and
                interpolator will work in these axes
            ycol (string): (optional) column name for dependent variable. Completeness grids and
                interpolator will work in these axes

        """
        self.recoveries = recoveries
        self.searches = searches

        if mstar is not None:
            self.mstar = np.zeros_like(self.recoveries['inj_period']) + mstar

            self.recoveries['inj_msini'] = radvel.utils.Msini(self.recoveries['inj_k'],
                                                              self.recoveries['inj_period'],
                                                              self.mstar, self.recoveries['inj_e'])
            self.recoveries['rec_msini'] = radvel.utils.Msini(self.recoveries['rec_k'],
                                                              self.recoveries['rec_period'],
                                                              self.mstar, self.recoveries['rec_e'])

            self.recoveries['inj_au'] = radvel.utils.semi_major_axis(self.recoveries['inj_period'], mstar)
            self.recoveries['rec_au'] = radvel.utils.semi_major_axis(self.recoveries['rec_period'], mstar)

            if teff is not None and rstar is not None: 
                self.recoveries['inj_sinc'] = rvsearch.utils.insolation(teff, rstar, self.recoveries['inj_au'])
                self.recoveries['rec_sinc'] = rvsearch.utils.insolation(teff, rstar, self.recoveries['rec_au'])

                self.recoveries['inj_teq'] = rvsearch.utils.tequil(teff, rstar, self.recoveries['inj_sinc'])
                self.recoveries['rec_teq'] = rvsearch.utils.tequil(teff, rstar, self.recoveries['rec_sinc'])

        self.xcol = xcol
        self.ycol = ycol

        self.grid = None
        self.interpolator = None
        
        #self.trends_count = trends_count

    @classmethod
    def from_csv(cls, recovery_file, *args, **kwargs):
        """Read recoveries and create Completeness object"""
        recoveries = pd.read_csv(recovery_file)

        return cls(recoveries, *args, **kwargs)

    def completeness_grid(self, xlim, ylim, resolution=30, xlogwin=0.5, ylogwin=0.5, trends_count=False):
        """Calculate completeness on a fine grid

        Compute a 2D moving average in loglog space

        Args:
            xlim (tuple): min and max x limits
            ylim (tuple): min and max y limits
            resolution (int): (optional) grid is sampled at this resolution
            xlogwin (float): (optional) x width of moving average
            ylogwin (float): (optional) y width of moving average

        """
        xgrid = np.logspace(np.log10(xlim[0]),
                            np.log10(xlim[1]),
                            resolution)
        ygrid = np.logspace(np.log10(ylim[0]),
                            np.log10(ylim[1]),
                            resolution)

        xinj = self.recoveries[self.xcol]
        yinj = self.recoveries[self.ycol]

        # If trends_count, then ONLY trends count. Treat resolved as non-detections.
        if trends_count:
            #good = self.recoveries[['recovered', 'trend_pref']].any(axis=1) # Is either one True?
            good = self.recoveries['trend_pref']
        else:
            good = self.recoveries['recovered']


        z = np.zeros((len(ygrid), len(xgrid)))
        last = 0
        for i,x in enumerate(xgrid):
            for j,y in enumerate(ygrid):
                xlow = 10**(np.log10(x) - xlogwin/2)
                xhigh = 10**(np.log10(x) + xlogwin/2)
                ylow = 10**(np.log10(y) - ylogwin/2)
                yhigh = 10**(np.log10(y) + ylogwin/2)

                xbox = yinj[np.where((xinj <= xhigh) & (xinj >= xlow))[0]]
                if len(xbox) == 0 or y > max(xbox) or y < min(xbox):
                    z[j, i] = np.nan
                    continue

                boxall = np.where((xinj <= xhigh) & (xinj >= xlow) &
                                  (yinj <= yhigh) & (yinj >= ylow))[0]
                boxgood = np.where((xinj[good] <= xhigh) &
                                   (xinj[good] >= xlow) & (yinj[good] <= yhigh) &
                                   (yinj[good] >= ylow))[0]
                # print(x, y, xlow, xhigh, ylow, yhigh, len(boxgood), len(boxall))
                if len(boxall) > 5:
                    z[j, i] = float(len(boxgood))/len(boxall)
                    last = float(len(boxgood))/len(boxall)
                else:
                    z[j, i] = np.nan

        self.grid = (xgrid, ygrid, z)

        return (xgrid, ygrid, z)

    def interpolate(self, x, y, refresh=False):
        """Interpolate completeness surface

        Interpolate completeness surface at x, y. X, y should be in the same
        units as self.xcol and self.ycol

        Args:
            x (array): x points to interpolate to
            y (array): y points to interpolate to
            refresh (bool): (optional) refresh the interpolator?

        Returns:
            array : completeness value at x and y

        """
        if self.interpolator is None or refresh:
            assert self.grid is not None, "Must run Completeness.completeness_grid before interpolating."
            gi = rvsearch.utils.cartesian_product(self.grid[0], self.grid[1])
            zi = self.grid[2].T
            self.interpolator = RegularGridInterpolator((self.grid[0], self.grid[1]), zi)

        return self.interpolator((x, y))
