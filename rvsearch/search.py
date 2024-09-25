"""Search class"""

import os
import copy
import pdb
import pickle

import numpy as np
import matplotlib.pyplot as pl
import corner

from astropy.timeseries import LombScargle
import astropy.constants as c
import radvel
import radvel.fitting
from radvel.plot import orbit_plots

from radvel import driver
from radvel import utils as radvel_utils
from radvel.mcmc import statevars
import pandas as pd

import rvsearch.periodogram as periodogram
import rvsearch.utils as utils


class Search(object):
    """Class to initialize and modify posteriors as planet search runs.

    Args:
        data (DataFrame): pandas dataframe containing times, vel, err, and insts.
        post (radvel.Posterior): Optional posterior with known planet params.
        starname (str): String, used to name the output directory.
        max_planets (int): Integer, limit on iterative planet search.
        priors (list): List of radvel prior objects to use.
        crit (str): Either 'bic' or 'aic', depending on which criterion to use.
        fap (float): False-alarm-probability to pass to the periodogram object.
        min_per (float): Minimum search period, to pass to the periodogram object.
        trend (bool): Whether to perform a DBIC test to select a trend model.
        linear(bool): Wether to linearly optimize gamma offsets.
        fix (bool): Whether to fix known planet parameters during search.
        polish (bool): Whether to create finer period grid after planet is found.
        verbose (bool):
        save_outputs (bool): Save output plots and files? [default = True]
        mstar (tuple): (optional) stellar mass and uncertainty in solar units

    """
    def __init__(self, data, post=None, setup_path=None, starname='star', setup_name=None, max_planets=8,
                priors=[], crit='bic', fap=0.001, min_per=3, max_per=10000,
                jity=2., manual_grid=None, oversampling=1., trend=False, linear=True,
                eccentric=False, fix=False, polish=True, baseline=True, sub_trend=False,
                mcmc=True, workers=1, verbose=True, save_outputs=True, save_plots=True, mstar=None):
        if {'time', 'mnvel', 'errvel', 'tel'}.issubset(data.columns):
            self.data = data
            self.tels = np.unique(self.data['tel'].values)
        elif {'jd', 'mnvel', 'errvel', 'tel'}.issubset(data.columns):
            self.data = data
            self.data.time = self.data.jd
            self.tels = np.unique(self.data['tel'].values)
        else:
            raise ValueError('Incorrect data input.')

        self.starname = starname
        self.linear   = linear
        if mstar is not None:
            self.mstar     = mstar[0]
            self.mstar_err = mstar[1]
        else:
            self.mstar     = None
            self.mstar_err = None

        if jity is None:
            self.jity = np.std(data.mnvel)
        else:
            self.jity = jity

        if post == None:
            self.basebic = None
        else:
            self.basebic = post.likelihood.bic()

        if post == None:

            ## Judah addition so RVSearch can accept setup files
            if setup_path is not None:

                P, post = radvel_utils.initialize_posterior(setup_path)
                self.post = post

                self.priors = post.priors
                self.params = self.post.params
                self.setup = True
                self.setup_name = setup_name
                self.starname = P.starname
                self.time_base = P.time_base
                self.setup_planets = post.params.num_planets
                self.params_init = post.params
                #import pdb; pdb.set_trace()
                


            else:
                self.priors = priors
                self.params = utils.initialize_default_pars(instnames=self.tels,
                                                        times=data.time,
                                                        linear=self.linear,
                                                        jitty=self.jity)
                self.post   = utils.initialize_post(data, params=self.params,
                                                priors=self.priors,
                                                linear=self.linear)
                self.setup  = False
                self.setup_name = None
                self.setup_planets = -1
                self.time_base = None

        else:
            #self.post          = post
            self.params_init   = post.params
            self.priors        = post.priors
            self.setup         = True
            self.setup_name = setup_name
            self.setup_planets = post.params.num_planets
            self.time_base = post.likelihood.model.time_base # Judah. Preserve original time base.

            # Judah. Initialize posterior immediately to apply gamma offsets to data. Mostly aesthetic.
            self.post = utils.initialize_post(self.data, post.params, post.priors, time_base=self.time_base)

        #import pdb; pdb.set_trace()

        self.all_params = []

        self.max_planets = max_planets
        if self.post.params.num_planets == 1 and self.post.params['k1'].value == 0.:
            self.num_planets = 0
        else:
            self.num_planets = self.post.params.num_planets

        self.crit = crit

        self.fap = fap
        self.min_per = min_per
        self.max_per = max_per

        self.trend     = trend
        self.eccentric = eccentric
        self.fix       = fix
        self.polish    = polish
        self.baseline  = baseline
        self.mcmc      = mcmc

        self.manual_grid  = manual_grid
        self.oversampling = oversampling
        self.workers      = workers
        self.verbose      = verbose
        self.save_outputs = save_outputs
        self.save_plots = save_plots

        self.pers = None
        self.periodograms = dict()
        self.bic_threshes = dict()
        self.best_bics = dict()
        self.eFAPs = dict()
        

        #import pdb; pdb.set_trace()
        # Judah. Subtract off trend before running initial search or any injections.
        self.sub_trend = sub_trend
        print("SUBBING TREND?: ", sub_trend)
        if self.sub_trend:
            self.trend_subtractor()

        self.trend_and_planet_allowed = True # By default, permit both a planet and trend in the model.


        #self.trend_curv_flat = "flat" # Default value for trend_test, in case it isn't run

    def trend_test(self):
        """Perform zero-planet baseline fit, test for significant trend.

        """
       
        post1 = copy.deepcopy(self.post)
        # Fix all Keplerian parameters. K is zero, equivalent to no planet.
        post1.params['k1'].vary      = False
        post1.params['tc1'].vary     = False
        post1.params['per1'].vary    = False
        post1.params['secosw1'].vary = False
        post1.params['sesinw1'].vary = False
        post1.params['dvdt'].vary    = True
        post1.params['curv'].vary    = True

        post1 = radvel.fitting.maxlike_fitting(post1, verbose=False)

        trend_curve_bic = post1.likelihood.bic()

        # Test without curvature
        post2 = copy.deepcopy(post1)
        post2.params['curv'].value = 0.0
        post2.params['curv'].vary  = False

        post2 = radvel.fitting.maxlike_fitting(post2, verbose=False)

        trend_bic = post2.likelihood.bic()

        # Test without trend or curvature
        post3 = copy.deepcopy(post2)
        post3.params['dvdt'].value = 0.0
        post3.params['dvdt'].vary  = False
        post3.params['curv'].value = 0.0
        post3.params['curv'].vary  = False

        flat_bic = post3.likelihood.bic()
        #import pdb; pdb.set_trace()
        #print("trend bics", flat_bic, trend_bic, trend_curve_bic)#, self.data.mnvel.iloc[0])
        #print("trend bics", flat_bic, post3.params['dvdt'].value)
        #flat_bic = -100000000
        if (trend_bic < flat_bic - 5) or (trend_curve_bic < flat_bic - 5):
            if trend_curve_bic < trend_bic - 5:
                # Quadratic
                #print("Quadratic chosennnnnnnnnnnn")
                self.post.params['dvdt'].value = post1.params['dvdt'].value
                self.post.params['curv'].value = post1.params['curv'].value
                self.post.params['dvdt'].vary  = True
                self.post.params['curv'].vary  = True
                
                self.trend_pref = True
                #self.trend_curv_flat = "curv"
                self.trend_bic_diff = trend_curve_bic - flat_bic
            else:
                # Linear
                #print("Linear chosennnnnnnnnnnn")
                self.post.params['dvdt'].value = post2.params['dvdt'].value
                self.post.params['curv'].value = 0
                self.post.params['dvdt'].vary  = True
                self.post.params['curv'].vary  = False
                
                self.trend_pref = True
                #self.trend_curv_flat = "trend"
                self.trend_bic_diff = trend_bic - flat_bic
        else:
            # Flat
            #print("Flat chosen")
            self.post.params['dvdt'].value = 0
            self.post.params['curv'].value = 0
            self.post.params['dvdt'].vary  = False
            self.post.params['curv'].vary  = False
            
            self.trend_pref = False
            self.trend_bic_diff = 0

        #import pdb; pdb.set_trace()
        return

    def trend_vs_planet(self):
        """
        Determine whether a model with trend, planet, or both is best.
        Added by Judah to avoid the situation where trend_test says trend
          is preferred, then a planet gets added later, and the trend and
          planet share a single signal and give spurious results.
        """

        ## Move to specified basis to fix params.
        self.post.params.basis.to_any_basis(self.post.params, "per tc secosw sesinw k")
        num_planets = self.post.params.num_planets

        #import pdb; pdb.set_trace()

        # Save the initial vary state of trend and curv for later.
        # Eg, if trend varies but curv is fixed, then don't bother letting curv vary in tests.
        dvdt_vary = self.post.params['dvdt'].vary
        curv_vary = self.post.params['curv'].vary



        post1 = copy.deepcopy(self.post)
        # Remove the most recent planet to test if trend explains signal
        post1.params['k{}'.format(num_planets)].value = 0
        for param_name in ['per', 'tc', 'secosw', 'sesinw', 'k']:
            post1.params[param_name+'{}'.format(num_planets)].vary = False
        #import pdb; pdb.set_trace()
        post1.params['dvdt'].vary = True#dvdt_vary
        post1.params['curv'].vary = True#curv_vary

        post1 = radvel.fitting.maxlike_fitting(post1, verbose=False)
        trend_curv_bic = post1.likelihood.bic()
        #import pdb; pdb.set_trace()
        

       
        post2 = copy.deepcopy(self.post)
        # Remove the trend to test if most recent planet explains signal
        #post2.params['per{}'.format(num_planets)].vary = False # Hold period constant
        
        for param_name in ['tc', 'secosw', 'sesinw', 'k']:
            post2.params[param_name+'{}'.format(num_planets)].vary = True

            if not self.eccentric:
                post2.params['secosw{}'.format(num_planets)].vary = False
                post2.params['sesinw{}'.format(num_planets)].vary = False

        
        #import pdb; pdb.set_trace()
        post2.params['dvdt'].value = 0
        post2.params['dvdt'].vary = False
        post2.params['curv'].value = 0
        post2.params['curv'].vary = False

        
        post2 = radvel.fitting.maxlike_fitting(post2, verbose=False)
        

        planet_bic = post2.likelihood.bic()


        #print("trend_bic: ", trend_curv_bic)
        #print("planet_bic: ", planet_bic)


        ## If having a trend and curve is allowed, then check all 3 models
        if self.trend_and_planet_allowed:

            post3 = copy.deepcopy(self.post)

            # Free both trend and most recent planet parameters
            post3.params['per{}'.format(num_planets)].vary = True # Hold period constant
            for param_name in ['tc', 'secosw', 'sesinw', 'k']:
                post3.params[param_name+'{}'.format(num_planets)].vary = True

                if not self.eccentric:
                    post3.params['secosw{}'.format(num_planets)].vary = False
                    post3.params['sesinw{}'.format(num_planets)].vary = False

        
            post3.params['dvdt'].vary = dvdt_vary
            post3.params['curv'].vary = curv_vary

            post3 = radvel.fitting.maxlike_fitting(post3, verbose=False)
            planet_trend_bic = post3.likelihood.bic()
            #print("both_bic: ", planet_trend_bic)

        



            if (planet_trend_bic < planet_bic - 15) and (planet_trend_bic < trend_curv_bic - 15):
                #print("Trend and planet won")

                for k in post3.params.keys():
                    self.post.params[k] = post3.params[k] # Assign value and vary state to self.post
                result="trend_and_planet"


            # Assign the winning model's parameters to self.post
            elif planet_bic < trend_curv_bic - 15: # Planet has to win by a ΔΒΙC of 30
                #print("Planet won")
            
                for k in post2.params.keys():
                    self.post.params[k] = post2.params[k] # Assign value and vary state to self.post
                result="planet"
        
            else: # Otherwise assign trend params to self.post
                #print("Trend won")
                for k in post1.params.keys():
                    self.post.params[k] = post1.params[k]
                self.sub_planet()
                self.num_planets -= 1 # Decrement num_planets because it was incremented by the detection
                result="trend"

        ## If having a trend and curve is NOT allowed, then check only 2 models
        else:
            # Assign the winning model's parameters to self.post
            if planet_bic < trend_curv_bic - 15:
                #print("Planet won")
            
                for k in post2.params.keys():
                    self.post.params[k] = post2.params[k] # Assign value and vary state to self.post
                result="planet"
        
            else: # Otherwise assign trend params to self.post
                #print("Trend won")
                for k in post1.params.keys():
                    self.post.params[k] = post1.params[k]
                self.sub_planet()
                self.num_planets -= 1 # Decrement num_planets because it was incremented by the detection
                result="trend"
                   



        #print("Trend, planet", trend_curv_bic, planet_bic, planet_trend_bic)
        #import pdb; pdb.set_trace()
        plott=False
        if plott:
            import matplotlib.pyplot as plt
            plt.close()

            model_y = post1.likelihood.model(post1.likelihood.x) + post1.params['gamma_j'].value    
            plt.scatter(post1.likelihood.x, post1.likelihood.y, c='r', label='data')
            plt.scatter(post1.likelihood.x, model_y, c='b', label='model')
            plt.legend()
            plt.savefig("post1_trend.png")
            plt.close()
            #print("p1", post1.params['dvdt'].value, post1.params['dvdt'].vary, post1.params['curv'].vary,  post1.params['k2'].value)

            data_x = post2.likelihood.x
            data_y = post2.likelihood.y

            model_x = np.linspace(np.min(data_x), np.max(data_x), 200)
            model_y = post2.likelihood.model(model_x) + post2.params['gamma_j'].value 
   
            plt.scatter(data_x, data_y, c='r', label='data')
            plt.scatter(model_x, model_y, c='b', label='model')
            plt.legend()
            plt.savefig("post2_planet.png")
            plt.close()
            #print("p2", post2.params['dvdt'].value, post2.params['dvdt'].vary, post2.params['curv'].vary, post2.params['k2'].value)

            if self.trend_and_planet_allowed:
                data_x = post3.likelihood.x
                data_y = post3.likelihood.y

                model_x = np.linspace(np.min(data_x), np.max(data_x), 200)
                model_y = post3.likelihood.model(model_x) + post3.params['gamma_j'].value
   
                plt.scatter(data_x, data_y, c='r', label='data')
                plt.scatter(model_x, model_y, c='b', label='model')
                plt.legend()
                plt.savefig("post3.png")
                plt.close()
            #import pdb; pdb.set_trace()
            #print("p3", post3.params['dvdt'].value, post3.params['dvdt'].vary, post3.params['curv'].vary, post3.params['k2'].value)



        
        ## Revert to synthesis basis to avoid mixing bases.
        self.post.params.basis.to_any_basis(self.post.params, "per tp e w k")

        return result



    def add_planet(self):
        """Add parameters for one more planet to posterior.

        """


        current_num_planets = self.post.params.num_planets
        fitting_basis = self.post.params.basis.name
        param_list = fitting_basis.split()

        new_num_planets = current_num_planets + 1

        default_pars = utils.initialize_default_pars(instnames=self.tels,
                                                     jitty=self.jity,
                                                     fitting_basis=fitting_basis)
        new_params = radvel.Parameters(new_num_planets, basis=fitting_basis)
                
        for planet in np.arange(1, new_num_planets):
            for par in param_list:
                parkey = par + str(planet)
                new_params[parkey] = self.post.params[parkey]

        
        for par in self.post.likelihood.extra_params:
            new_params[par] = self.post.params[par]  # For gamma and jitter

        
        # Set default parameters for n+1th planet
        default_params = utils.initialize_default_pars(self.tels,
                                                       jitty=self.jity,
                                                       fitting_basis=fitting_basis)

        for par in param_list:
            parkey = par + str(new_num_planets)
            onepar = par + '1'  # MESSY, FIX THIS 10/22/18
            new_params[parkey] = default_params[onepar]
        
        new_params['dvdt'] = self.post.params['dvdt']
        new_params['curv'] = self.post.params['curv']

        if not self.post.params['dvdt'].vary:
            new_params['dvdt'].vary = False
        if not self.post.params['curv'].vary:
            new_params['curv'].vary = False


        new_params['per{}'.format(new_num_planets)].vary = False
        if not self.eccentric:
            # Convert to basis containing secosw and sesinw
            new_params = new_params.basis.to_any_basis(new_params, 'per tc secosw sesinw k')

            new_params['secosw{}'.format(new_num_planets)].vary = False
            new_params['sesinw{}'.format(new_num_planets)].vary = False

            # Convert back to fitting basis
            new_params = new_params.basis.to_any_basis(new_params, fitting_basis)
        new_params.num_planets = new_num_planets

        # Respect setup file priors.
        if self.setup:
            priors = self.priors
        else:
            priors = []

        
        # Judah addition: Params that should have been listed in extra_params in the setup file, eg GP params
        forgotten_params = [prm for prm in self.post.list_params() if prm not in new_params.keys()]
        for prm in forgotten_params:
            new_params[prm] = self.post.params[prm]


        priors.append(radvel.prior.EccentricityPrior(new_num_planets)) # Compiles with each detection. NBD though
        #priors.append(radvel.prior.HardBounds("k{}".format(new_num_planets), 0, 100)) # Judah experiment for T001694

        new_post = utils.initialize_post(self.data, new_params, priors, time_base=self.time_base)
        self.post = new_post
        #print("Checking planet nums!", self.post.params.num_planets, self.num_planets)


    def sub_planet(self):
        """Remove parameters for one planet from posterior.

        """
        current_num_planets = self.post.params.num_planets
        fitting_basis = self.post.params.basis.name
        param_list = fitting_basis.split()
        num2letter = {1:'b', 2:'c', 3:'d', 4:'e',
                      5:'f', 6:'g', 7:'h', 8:'i'}
        current_max_letter = num2letter[current_num_planets] # Assign letter of planet to be dropped. Used in choosing which priors to drop below
        #import pdb; pdb.set_trace()
        new_num_planets = current_num_planets - 1

        default_pars = utils.initialize_default_pars(instnames=self.tels, jitty=self.jity)
        new_params = radvel.Parameters(new_num_planets, basis=fitting_basis)

        for planet in np.arange(1, new_num_planets+1):
            for par in param_list:
                parkey = par + str(planet)
                new_params[parkey] = self.post.params[parkey]


        for par in self.post.likelihood.extra_params:
            new_params[par] = self.post.params[par]

        new_params['dvdt'] = self.post.params['dvdt']
        new_params['curv'] = self.post.params['curv']

        if not self.post.params['dvdt'].vary:
            new_params['dvdt'].vary = False
        if not self.post.params['curv'].vary:
            new_params['curv'].vary = False

        
        ## Judah: drop any priors having to do specifically with dropped planet, but keep the rest
        priors = []
        for prior in self.post.priors:
            prior_str = prior.__str__()
            test_str = "{"+current_max_letter+"}"
            if test_str in prior_str:
                continue
            else:
                priors.append(prior)


        # Judah addition: Params that should have been listed in extra_params in the setup file, eg GP params. Needed for eg T001759, otherwise choose different setup file.
        forgotten_params = [prm for prm in self.post.list_params() if (prm not in new_params.keys() 
                                                                       and "gp" in prm)]
        for prm in forgotten_params:
            new_params[prm] = self.post.params[prm]

        new_post = utils.initialize_post(self.data, new_params, priors, time_base=self.time_base)
        self.post = new_post
        #import pdb; pdb.set_trace()

    def trend_subtractor(self):
        """
        Judah addition: subtract off trend/curv before any planet searching and fix it to 0.
        """

        if self.post == None:
            print("trend_subtractor: No post provided. Returning.")
            return
        elif self.post.params['dvdt'].value==0 and self.post.params['curv'].value==0:
            print("trend_subtractor: No trend/curv to subtract. Returning.")
            return
        else:
            print("trend_subtractor: Non-zero trend/curv detected. Subtracting!")
            plot_= False
            if plot_:
                print("PLOTTING in trend_subtractor")
                ## First, set all planet signals to K=0 except the one we're plotting
                #for i in range(self.post.params.num_planets-1):
                 #   k_key = "k{}".format(i+1)
                  #  self.post.likelihood.params[k_key].value=0

                import matplotlib.pyplot as plt
                plt.close()
 
                colors = ['green', 'red', 'purple', 'orange']
                for i, tel_name in enumerate(self.tels):
                    inds = np.where(self.post.likelihood.telvec==tel_name)

                    gamma_key = 'gamma_{}'.format(tel_name)
                    #import pdb; pdb.set_trace()
                    data_x = self.post.likelihood.x[inds]
                    data_y = self.post.likelihood.y[inds]
                    #data_x = self.data.query("tel=='{}'".format(tel_name)).time
                    #data_y = self.data.query("tel=='{}'".format(tel_name)).mnvel
                    color = colors[i]
                    plt.scatter(data_x, data_y, c=color, label='{}_before'.format(tel_name))
                

                data_time_full = self.post.likelihood.x
                model_x = np.linspace(np.min(data_time_full), np.max(data_time_full), 400)
                model_y = self.post.likelihood.model(model_x) #+ self.post.params['gamma_j'].value

                plt.plot(model_x, model_y, label='before_model')




            dvdt_signal = self.post.params['dvdt'].value*(self.data.time-self.post.model.time_base)
            curv_signal = self.post.params['curv'].value*(self.data.time-self.post.model.time_base)**2
            #import pdb; pdb.set_trace()
            self.data['mnvel'] -= (dvdt_signal+curv_signal)
            #self.post.likelihood.y -= (dvdt_signal+curv_signal)

            # Retain the subtracted values to be added back to the model later
            #import pdb; pdb.set_trace()
            self.subtracted_trend = copy.deepcopy(self.post.params['dvdt'])
            self.subtracted_curv = copy.deepcopy(self.post.params['curv'])

            self.post.params['dvdt'].value=0
            self.post.params['dvdt'].vary=False
            self.post.params['curv'].value=0
            self.post.params['curv'].vary=False

            

            self.post = utils.initialize_post(self.data, self.post.params, self.post.priors, time_base=self.time_base)

            if plot_:

                for i, tel_name in enumerate(self.tels):
                    inds = np.where(self.post.likelihood.telvec==tel_name)

                    data_x = self.post.likelihood.x[inds]
                    data_y = self.post.likelihood.y[inds]
                    color = colors[i]
                    plt.scatter(data_x, data_y, edgecolor=color, marker="s", facecolor="None", label='{}_after'.format(tel_name))

                model_x = np.linspace(np.min(data_x), np.max(data_x), 200)
                model_y = self.post.likelihood.model(model_x) #+ self.post.params['gamma_j'].value

                plt.plot(model_x, model_y, label='after_model')
                plt.legend()
                plt.savefig("/home/judahvz/my_papers/distant_giants/rvsearch_work/trend_sub.png")
                plt.close()


         #return self.post



    def fit_orbit(self):
        """Perform a max-likelihood fit with all parameters free.

        """
        #import pdb; pdb.set_trace()
        for n in np.arange(1, self.num_planets+1):
            # Respect setup planet fixed eccentricity and period.
            if n <= self.setup_planets:
                self.post.params['k{}'.format(n)].vary = self.params_init['k{}'.format(n)].vary
                self.post.params['tc{}'.format(n)].vary = self.params_init['tc{}'.format(n)].vary
                self.post.params['per{}'.format(n)].vary = self.params_init['per{}'.format(n)].vary
                self.post.params['secosw{}'.format(n)].vary = self.params_init['secosw{}'.format(n)].vary
                self.post.params['sesinw{}'.format(n)].vary = self.params_init['sesinw{}'.format(n)].vary

            else:
                self.post.params['k{}'.format(n)].vary = True
                self.post.params['tc{}'.format(n)].vary = True
                self.post.params['per{}'.format(n)].vary = True
                self.post.params['secosw{}'.format(n)].vary = True
                self.post.params['sesinw{}'.format(n)].vary = True

                if not self.eccentric:
                    self.post.params['secosw{}'.format(n)].vary = False
                    self.post.params['sesinw{}'.format(n)].vary = False
                    self.post.params['secosw{}'.format(n)].value = 0
                    self.post.params['sesinw{}'.format(n)].value = 0

        #import pdb; pdb.set_trace()
        self.polish=True
        if self.polish:
            # Make a finer, narrow period grid, and search with eccentricity.
            self.post.params['per{}'.format(self.num_planets)].vary = False
            default_pdict = {}
            for k in self.post.params.keys():
                default_pdict[k] = self.post.params[k].value
            polish_params = []
            polish_bics = []
            #import pdb; pdb.set_trace()

            peak = np.argmax(self.periodograms[self.num_planets-1])
            if self.manual_grid is not None:
                # Polish around 1% of period value if manual grid specified
                # especially useful in the case that len(manual_grid) == 1
                subgrid = np.linspace(0.99*self.manual_grid[peak], 1.01*self.manual_grid[peak], 9)


            elif peak == len(self.periodograms[self.num_planets-1]) - 1: # If the best period is the longest one
                subgrid = np.linspace(self.pers[peak-1], 2*self.pers[peak] - self.pers[peak-1], 9)
            else:  # TO-DO: JUSTIFY 9 GRID POINTS, OR TAKE AS ARGUMENT
                subgrid = np.linspace(self.pers[peak-1], self.pers[peak+1], 9)

            fit_params = []
            power = []
            
            for per in subgrid:
                for k in default_pdict.keys():
                    self.post.params[k].value = default_pdict[k]
                perkey = 'per{}'.format(self.num_planets)
                self.post.params[perkey].value = per
                self.post.params[perkey].vary = False # Judah: Force period to stay there?

                fit = radvel.fitting.maxlike_fitting(self.post, verbose=False)
                power.append(-fit.likelihood.bic())

                best_params = {}
                for k in fit.params.keys():
                    best_params[k] = fit.params[k].value
                fit_params.append(best_params)
                
            
            fit_index = np.argmax(power)
            bestfit_params = fit_params[fit_index]

            for k in self.post.params.keys():
                self.post.params[k].value = bestfit_params[k]

            
            trend_or_planet = self.trend_vs_planet() # "trend", "planet", or "trend_and_planet"
            
            if trend_or_planet in ["planet", "trend_and_planet"]:
                planetary=True
                # Only let the most recent period vary if a planet was found. If it's a trend, don't.
                self.post.params['per{}'.format(self.num_planets)].vary = True
            elif trend_or_planet=="trend":
                planetary=False
        #print("Done with polish")

        
        plott=False
        if plott:
            import matplotlib.pyplot as plt
            plt.close()
            data_x = self.post.likelihood.x
            data_y = self.post.likelihood.y
            model_x = np.linspace(np.min(data_x), np.max(data_x), 200)
            model_y = self.post.likelihood.model(model_x)+ self.post.params['gamma_j'].value

            plt.scatter(data_x, data_y, c='b', label='data')
            plt.plot(model_x, model_y, c='b', label='model_before')
            
        self.post = radvel.fitting.maxlike_fitting(self.post, verbose=False)

        if plott:
            model_x = np.linspace(np.min(data_x), np.max(data_x), 200)
            model_y = self.post.likelihood.model(model_x)+ self.post.params['gamma_j'].value

            plt.plot(model_x, model_y, c='r', label='after_model')
            plt.legend()
            plt.savefig("fit_orbit_before_after.png")
            plt.close()


        if self.fix:
            for n in np.arange(1, self.num_planets+1):
                self.post.params['per{}'.format(n)].vary = False
                self.post.params['k{}'.format(n)].vary = False
                self.post.params['tc{}'.format(n)].vary = False
                self.post.params['secosw{}'.format(n)].vary = False
                self.post.params['sesinw{}'.format(n)].vary = False

        return planetary

    def save(self, filename='post_final.pkl'):
        """Pickle current posterior.

        """
        self.post.writeto(filename)

    def running_per(self):
        """Generate running BIC periodograms for each planet/signal.

        """
        nobs = len(self.post.likelihood.x)
        # Sort times, RVs, and RV errors chronologically.
        indices = np.argsort(self.post.likelihood.x)
        x    = self.post.likelihood.x[indices]
        y    = self.post.likelihood.y[indices]
        yerr = self.post.likelihood.yerr[indices]
        tels = self.post.likelihood.telvec[indices]
        # Generalized Lomb-Scargle version; functional, but seems iffy.
        # Subtract off gammas and trend terms.
        for tel in self.tels:
            y[np.where(tels == tel)] -= self.post.params['gamma_{}'.format(tel)].value

        if self.post.params['dvdt'].vary == True:
            y -= self.post.params['dvdt'].value * (x - self.post.likelihood.model.time_base)

        if self.post.params['curv'].vary == True:
            y -= self.post.params['curv'].value * (x - self.post.likelihood.model.time_base)**2

        # Instantiate a list to populate with running periodograms.
        runners = []
        # Iterate over the planets/signals.
        
        for n in np.arange(1, self.num_planets+1):
            runner = []
            planets = np.arange(1, self.num_planets+1)
            yres = copy.deepcopy(y)
            self.post.params = self.post.params.basis.to_synth(self.post.params)
            for p in planets[planets != n]:
                orbel = [self.post.params['per{}'.format(p)].value,
                         self.post.params['tp{}'.format(p)].value,
                         self.post.params['e{}'.format(p)].value,
                         self.post.params['w{}'.format(p)].value,
                         self.post.params['k{}'.format(p)].value]
                yres -= radvel.kepler.rv_drive(x, orbel)
            # Make small period grid. Figure out proper spacing.
            per = self.post.params['per{}'.format(n)].value
            subpers1 = np.linspace(0.95*per, per, num=99, endpoint=False)
            subpers2 = np.linspace(per, 1.05*per, num=100)
            subpers  = np.concatenate((subpers1, subpers2))

            for i in np.arange(12, nobs+1):
                freqs = 1. / subpers
                power = LombScargle(x[:i], yres[:i], yerr[:i],
                                    normalization='psd').power(freqs)
                runner.append(np.amax(power))

            runners.append(runner)
        self.runners = runners

    def run_search(self, fixed_threshold=None, outdir=None, mkoutdir=True,
                   running=True):
        """Run an iterative search for planets not given in posterior.

        Args:
            fixed_threshold (float): (optional) use a fixed delta BIC threshold
            mkoutdir (bool): create the output directory?
        """

        if outdir is None:
            outdir = os.path.join(os.getcwd(), self.starname)
        if mkoutdir and not os.path.exists(outdir):
            os.mkdir(outdir)


        # Judah: let jitter float. If a planet was missed in the initial radvel run, jitter will be too big
        for key in self.post.params.keys():
           if "jit" in key:
               self.post.params[key].vary = True

        #import pdb; pdb.set_trace()
        
 
        if self.trend:
            self.trend_test()

        run = True


        #print("Min period", self.min_per)

        
        while run:

            ## Judah: put this check before search to avoid unnecessary search
            if self.num_planets >= self.max_planets:
                run = False

            if self.num_planets != 0:
                ## NOTE: add_planet() re-initializes the posterior
                #import pdb; pdb.set_trace()
                self.add_planet()

            #print("Are we doing the search?", self.num_planets, self.max_planets, run)

            ## Judah test: new per grid for sparsely sampled systems
            #min_per=100
            #max_per=np.max(self.pers)
            #man_grid_len = int(max(len(self.pers)/4, 50))
            #self.manual_grid = 1/np.linspace(1/min_per, 1/max_per, man_grid_len)
            #self.pers = self.manual_grid

            #import pdb; pdb.set_trace()
            perioder = periodogram.Periodogram(self.post, basebic=self.basebic,
                                               minsearchp=self.min_per,
                                               maxsearchp=self.max_per,
                                               fap=self.fap,
                                               manual_grid=self.manual_grid,
                                               oversampling=self.oversampling,
                                               baseline=self.baseline,
                                               eccentric=self.eccentric,
                                               workers=self.workers,
                                               verbose=self.verbose,
                                               starname=self.starname)
 

            perioder.per_bic()

            
            #print("BICs from perioder.bic()", perioder.bic)
            #sdfdsf
            #import pdb; pdb.set_trace()

            self.periodograms[self.num_planets] = perioder.power[self.crit]
            if self.num_planets == 0 or self.pers is None:
                self.pers = perioder.pers
            
            if fixed_threshold is None:
                perioder.eFAP()
                self.eFAPs[self.num_planets] = perioder.fap_min
            else:
                perioder.bic_thresh = fixed_threshold

            #print("Checking the thresh", fixed_threshold, perioder.bic_thresh, self.fap)

            self.bic_threshes[self.num_planets] = perioder.bic_thresh
            self.best_bics[self.num_planets] = perioder.best_bic
            # Judah: two of the best periods:
            self.ls_best_per = perioder.ls_best_per
            self.bic_best_per = perioder.bic_best_per
            #import pdb; pdb.set_trace()
            
            
            #print("BIC PER", self.bic_best_per)
            #print("Thresh and best", perioder.bic_thresh, perioder.best_bic)
            #import pdb; pdb.set_trace()

            
            if self.save_outputs:
                #print("Going in to save", self.num_planets)
                try:
                    perioder.plot_per()
                except:
                    import pdb; pdb.set_trace()
                perioder.fig.savefig(outdir+'/dbic{}.pdf'.format(
                                     self.num_planets+1))

            # Check whether there is a detection. If so, fit free and proceed.
            ## Judah experiment: allow pass if best_bic is within 0.1% of threshold
            #within_1percent = (abs(perioder.best_bic-perioder.bic_thresh)<0.001*perioder.bic_thresh)
            #if (perioder.best_bic < perioder.bic_thresh) and within_1percent:
             #   print("1 PERCENTER", perioder.bic_thresh, perioder.best_bic, self.bic_best_per)
            #if (perioder.best_bic > perioder.bic_thresh) or within_1percent:
            if perioder.best_bic > perioder.bic_thresh:
                self.num_planets += 1
                for k in self.post.params.keys():
                    self.post.params[k].value = perioder.bestfit_params[k]


                #import pdb; pdb.set_trace()
                # Generalize tc reset to each new discovery.
                tckey = 'tc{}'.format(self.num_planets)
                if self.post.params[tckey].value < np.amin(self.data.time):
                    self.post.params[tckey].value = np.median(self.data.time)
                    for n in np.arange(1, self.num_planets+1):
                        self.post.params['k{}'.format(n)].vary = False
                        self.post.params['per{}'.format(n)].vary = False
                        self.post.params['secosw{}'.format(n)].vary = False
                        self.post.params['sesinw{}'.format(n)].vary = False
                        if n != self.num_planets:
                            self.post.params['tc{}'.format(n)].vary = False
                    #import pdb; pdb.set_trace()
                    self.post = radvel.fitting.maxlike_fitting(self.post,
                                                               verbose=False)

                    for n in np.arange(1, self.num_planets+1):
                        self.post.params['k{}'.format(n)].vary = True
                        self.post.params['per{}'.format(n)].vary = True
                        self.post.params['secosw{}'.format(n)].vary = True
                        self.post.params['sesinw{}'.format(n)].vary = True
                        self.post.params['tc{}'.format(n)].vary = True


                #import pdb; pdb.set_trace()
                # Idea: if a signal is detected as a planet, but then turns out to be a trend, then finish the search. Otherwise when you go to look again, you will find the same trend and get caught in a loop. Alternative is to fix the trend where you find it, which I didn't try.
                run = self.fit_orbit() # Judah add. Make fit_orbit return T/F depending on trend_test.

                self.all_params.append(self.post.params)
                self.basebic = self.post.likelihood.bic()
                #import pdb; pdb.set_trace()

               
            
            else:
                #import pdb; pdb.set_trace()
                self.sub_planet()
                # 8/3: Update the basebic anyway, for injections.
                self.basebic = self.post.likelihood.bic()
                run = False
            

            if self.num_planets >= self.max_planets:
                run = False


            # If any jitter values are negative, flip them.
            for key in self.post.params.keys():
                if 'jit' in key:
                    if self.post.params[key].value < 0:
                        self.post.params[key].value = -self.post.params[key].value
            # Generate an orbit plot.
            if self.save_plots:
                rvplot = orbit_plots.MultipanelPlot(self.post, saveplot=outdir +
                                                    self.num_planets)
                multiplot_fig, ax_list = rvplot.plot_multipanel()
                multiplot_fig.savefig(outdir+'/orbit_plot{}.pdf'.format(
                                                        self.num_planets))

            #import pdb; pdb.set_trace()

        # Generate running periodograms.
        if running:
            self.running_per()

     


        # Run MCMC on final posterior, save new parameters and uncertainties
        #import pdb; pdb.set_trace()
        if self.mcmc == True and (self.num_planets != 0 or
                                  self.post.params['dvdt'].vary == True):
            self.post.uparams   = {}
            self.post.medparams = {}
            self.post.maxparams = {}
            
            # Default to no plots in report
            corner_plot = False
            multipanel_plot = False


            # Use recommended parameters for mcmc.
            nensembles = np.min([self.workers, 16])

            if os.cpu_count() < nensembles:
                nensembles = os.cpu_count()
            # Set custom mcmc scales for e/w parameters.
            for n in np.arange(1, self.num_planets+1):
                self.post.params['secosw{}'.format(n)].mcmcscale = 0.005
                self.post.params['sesinw{}'.format(n)].mcmcscale = 0.005


            # Sample in log-period space.
            logpost = copy.deepcopy(self.post)
            logparams = logpost.params.basis.to_any_basis(
                        logpost.params, 'logper tc secosw sesinw k')

            #import pdb; pdb.set_trace()
            keep_priors = []
            drop_strs = ["{b}", "{c}", "{d}", "{e}", "{f}", "{g}", "{h}", "{i}"]

            # Keep priors that do not pertain to a specific planet. Also keep e<0.99 priors.
            # I don't remember why I remove other planet-specific priors. Scrutinize more.
            for prior in logpost.priors:
                prior_str = prior.__str__()

                if "constrained to be $<0.99$" in prior_str: # Keep e<0.99
                    keep_priors.append(prior)

                drop_str_in_prior = [drop_str in prior_str for drop_str in drop_strs]
                if any(drop_str_in_prior): # Drop other planet-specific
                    continue
                else:
                    keep_priors.append(prior)
            logpost = utils.initialize_post(self.data, params=logparams, priors=keep_priors) # Judah: This is where logpost.model.time_base becomes different from self.post.model.time_base. I reset it below.

            ## Priors to keep jitter from blowing up
            for like in self.post.likelihood.like_list: # Each telescope has its own likelihood in like_list
                jit_param = like.jit_param
 
                logpost.priors = logpost.priors + [radvel.prior.Gaussian(jit_param, 0, 5)]
                logpost.priors = logpost.priors + [radvel.prior.HardBounds(jit_param, 1, 15)]



            for n in np.arange(1, logpost.params.num_planets+1):
                
                if n<=self.setup_planets:
                    logpost.params["logper{}".format(n)].vary = True # Let known planet pers vary (should have priors)
                    logpost.params["k{}".format(n)].vary = True # Refit all known K-amplitudes

                    # Force transiting planets to circular. Rare reduction in accuracy, but usually helps good fits.
                    logpost.params["secosw{}".format(n)].value = 0
                    logpost.params["sesinw{}".format(n)].value = 0
                    logpost.params["secosw{}".format(n)].vary = False
                    logpost.params["sesinw{}".format(n)].vary = False


                # Judah (for inner planets): if planet has gaussian period prior, set 3sig logper bounds
                has_per_prior=False
                for prior in self.post.priors:
                    if prior.__str__().split(' ')[0]=="Gaussian": # If it's a Gaussian prior
                        if prior.param=="per{}".format(n):
                            if prior.mu/prior.sigma>20: # If the period is a <20sigma detection, vary over 20 sigma
                                sig_fac = 20
                            else: # Otherwise, vary over only 10 sigma
                                sig_fac=10
                            min_bound = np.log(prior.mu-sig_fac*prior.sigma)
                            max_bound = np.log(prior.mu+sig_fac*prior.sigma)
                            
                            per_prior = radvel.prior.HardBounds('logper{}'.format(n), min_bound, max_bound)
                            logpost.priors = logpost.priors + [per_prior]
                            has_per_prior = True
                        
                if has_per_prior==False: # If no gaussian prior, set broad limits: 0.0067 - 22k days
                    logpost.priors = logpost.priors + [radvel.prior.HardBounds('logper{}'.format(n), -5, 10.0)]            

                #if n>self.setup_planets:
                ecos_sq = logpost.params['secosw{}'.format(n)].value**2
                esin_sq = logpost.params['sesinw{}'.format(n)].value**2
                e = ecos_sq + esin_sq

                print("THIS IS e{}: {}".format(n, e))
                if (e<0.1 or e>0.95):
                    logpost.params['secosw{}'.format(n)].value = 0
                    logpost.params['sesinw{}'.format(n)].value = 0

                    logpost.params['secosw{}'.format(n)].vary = False
                    logpost.params['sesinw{}'.format(n)].vary = False
        

            #import pdb; pdb.set_trace()
            logpost_ = copy.deepcopy(logpost)


            # Run MCMC.
            # Judah: define some mcmc params for .stat file below
            nsteps = 4000
            minafactor = 15
            maxarchange = 0.07
            mintz = 2000
            maxgr = 1.0075


            logpost.model.time_base = self.post.model.time_base
            

            chains = radvel.mcmc(logpost, nwalkers=50, nrun=nsteps, burnGR=1.03,
                                 maxGR=maxgr, minTz=mintz, minAfactor=minafactor,
                                 maxArchange=maxarchange, burnAfactor=15,
                                 minsteps=12500, minpercent=5, thin=5,
                                 save=False, ensembles=nensembles)

            #import pdb; pdb.set_trace()
            # Convert chains to per, e, w basis.
            # Judah: first add columns to chains for the vary=False parameters so they show up in report
            # Note: mcmc() above CHANGES logpost params, even if fixed. Use logpost_ to retrieve fixed param values and place them in chains
            for par in logpost_.list_params():
                if par not in chains.columns:
                    chains[par] = np.zeros(len(chains)) + logpost_.params[par].value # Column of single value
            #import pdb; pdb.set_trace()
            synthchains = logpost.params.basis.to_synth(chains)
            synthquants = synthchains.quantile([0.159, 0.5, 0.841])

            # Compress, thin, and save chains, in fitting and synthetic bases.
            csvfn = outdir + '/chains.csv.tar.bz2'
            synthchains.to_csv(csvfn, compression='bz2')

            # Retrieve e and w medians & uncertainties from synthetic chains.
            for n in np.arange(1, self.num_planets+1):
                e_key = 'e{}'.format(n)
                w_key = 'w{}'.format(n)
                # Add period if it's a synthetic parameter.
                per_key = 'per{}'.format(n)
                logper_key = 'logper{}'.format(n)
                k_key = 'k{}'.format(n)

                med_e  = synthquants[e_key][0.5]
                high_e = synthquants[e_key][0.841] - med_e
                low_e  = med_e - synthquants[e_key][0.159]
                err_e  = np.mean([high_e,low_e])
                err_e  = radvel.utils.round_sig(err_e)
                med_e, err_e, errhigh_e = radvel.utils.sigfig(med_e, err_e)
                max_e, err_e, errhigh_e = radvel.utils.sigfig(
                                          self.post.params[e_key].value, err_e)

                med_w  = synthquants[w_key][0.5]
                high_w = synthquants[w_key][0.841] - med_w
                low_w  = med_w - synthquants[w_key][0.159]
                err_w  = np.mean([high_w,low_w])
                err_w  = radvel.utils.round_sig(err_w)
                med_w, err_w, errhigh_w = radvel.utils.sigfig(med_w, err_w)
                max_w, err_w, errhigh_w = radvel.utils.sigfig(
                                          self.post.params[w_key].value, err_w)

                self.post.uparams[e_key]   = err_e
                self.post.uparams[w_key]   = err_w
                self.post.medparams[e_key] = med_e
                self.post.medparams[w_key] = med_w
                self.post.maxparams[e_key] = max_e
                self.post.maxparams[w_key] = max_w

            # Retrieve medians & uncertainties for the fitting basis parameters.
            for par in self.post.params.keys():
                #if self.post.params[par].vary: # Only vary params. Else fixed param sigfigs get messed up
                med = synthquants[par][0.5]
                high = synthquants[par][0.841] - med
                low = med - synthquants[par][0.159]
                err = np.mean([high,low])
                err = radvel.utils.round_sig(err)
                med, err, errhigh = radvel.utils.sigfig(med, err)
                max_, err, errhigh = radvel.utils.sigfig(
                                self.post.params[par].value, err)

                self.post.uparams[par] = err
                self.post.medparams[par] = med
                self.post.maxparams[par] = max_

                ## Judah: set post param values to new values. Fixes plots
                self.post.params[par].value = synthquants[par][0.5]

            ## Judah: also set new model param values so they show up in plot.
            self.post.model.params = self.post.params 
            #import pdb; pdb.set_trace()


        #import pdb; pdb.set_trace()

        if self.save_outputs:
            chain_path = outdir + '/chains.csv.tar.bz2'
            synthchains = pd.read_csv(chain_path)
            #import pdb; pdb.set_trace()

            ## Use RVSearch's derive function to add derived parameters to posterior
            if self.mstar is not None:
                self.post, synthchains = utils.derive(self.post, synthchains, self.mstar, self.mstar_err)

            post_path = os.path.join(outdir, "post_final.pkl")
            
            # Judah addition: save summary csv (code taken from radvel.driver.mcmc())
            if (self.num_planets != 0 or self.post.params['dvdt'].vary == True):
                post_summary = chains.quantile([0.159, 0.5, 0.841])
                saveto = os.path.join(outdir, self.starname+"_post_summary.csv")
                post_summary.to_csv(saveto, sep=",")

                report = True

                if report:

                    #import pdb; pdb.set_trace()

                    # Judah: initialize statfile
                    statfile = os.path.join(outdir, '{}_radvel.stat'.format(self.setup_name))
                    status = driver.load_status(statfile)

                    setup_path = "/home/judahvz/my_papers/trends_paper/rv/rv_data/{}/{}.py".format(self.starname, self.setup_name)
                    
                    chain_path = os.path.join(outdir, "chains.csv.tar.bz2")
                    autocorr = os.path.join(outdir, self.setup_name+"_autocorr.csv")

                    mcmc_savestate = {'run': True,
                                      'postfile': os.path.relpath(post_path),
                                      'chainfile': os.path.relpath(chain_path),
                                      'autocorrfile': os.path.relpath(autocorr),
                                      'summaryfile': os.path.relpath(saveto),
                                      'nwalkers': statevars.nwalkers,
                                      'nensembles': nensembles,
                                      'maxsteps': nsteps*statevars.nwalkers*nensembles,
                                      'nsteps': statevars.ncomplete,
                                      'nburn': statevars.nburn,
                                      'minafactor': minafactor,
                                      'maxarchange': maxarchange,
                                      'minTz': mintz,
                                      'maxGR': maxgr}
                    driver.save_status(statfile, 'mcmc', mcmc_savestate)


                    #import pdb; pdb.set_trace()
                    P, initial_post = radvel_utils.initialize_posterior(setup_path) # Initialize to get P. Forget about initial_post
                    #import pdb; pdb.set_trace()
                    #post = radvel.posterior.load(post_path) # Judah: use RVSearch post, not initialized from setup file
                    #chains = pd.read_csv(chain_path)
                    P.nplanets = self.post.params.num_planets # Judah. Match setup file nplanets to post nplanets
                    P.planet_letters = {1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i'} # Judah: provide planet letters in case more planets were found than just what was listed in the setup file



                    # Judah: Use radvel's derive functions to save derived params in stat file
                    

                    status = driver.load_status(statfile)
                    """
                    try:
                        mstar = np.random.normal(
                                     loc=P.stellar['mstar'], scale=P.stellar['mstar_err'],
                                     size=len(chains)
                                                 )
                    except AttributeError:
                        print("Mstar not provided in radvel setup file. Trying value passed to RVSearch.")
                        if self.mstar is not None:
                            mstar = np.random.normal(
                                     loc=self.mstar[0], scale=self.mstar[1],
                                     size=len(chains)
                                                 )
                        else:
                             raise Exception("Err: Must provide a stellar mass")
                             


                    if (self.mstar <= 0.0).any():
                        num_nan = np.sum(self.mstar <= 0.0)
                        nan_perc = float(num_nan) / len(chains)
                        mstar[mstar <= 0] = np.abs(mstar[mstar <= 0])
                        print("WARNING: {} ({:.2f} %) of Msini samples are NaN. The stellar mass posterior may contain negative \
values. Interpret posterior with caution.".format(num_nan, nan_perc))


                    synthchains = post.params.basis.to_synth(synthchains)

                    savestate = {'run': True}
                    
                    outcols = []
                    for i in np.arange(1, P.nplanets + 1, 1):
                        # Grab parameters from the chain
                        def _has_col(key):
                            cols = list(synthchains.columns)
                            return cols.count('{}{}'.format(key, i)) == 1

                        def _get_param(key):
                            if _has_col(key):
                                return synthchains['{}{}'.format(key, i)]
                            else:
                                return P.params['{}{}'.format(key, i)].value

                        def _set_param(key, value):
                            chains['{}{}'.format(key, i)] = value

                        def _get_colname(key):
                            return '{}{}'.format(key, i)

                        per = _get_param('per')
                        k = _get_param('k')
                        e = _get_param('e')
                        #import pdb; pdb.set_trace()
                        mpsini = radvel.utils.Msini(k, per, mstar, e, Msini_units='earth')
                        _set_param('mpsini', mpsini)
                        outcols.append(_get_colname('mpsini'))

                        mtotal = mstar + (mpsini * c.M_earth.value) / c.M_sun.value   # get total star plus planet mass
                        a = radvel.utils.semi_major_axis(per, mtotal)         # changed from mstar to mtotal
        
                        _set_param('a', a)
                        outcols.append(_get_colname('a'))

                        musini = (mpsini * c.M_earth.value) / (mstar * c.M_sun.value)
                        _set_param('musini', musini)
                        outcols.append(_get_colname('musini'))

                        try:
                            rp = np.random.normal(
                                loc=P.planet['rp{}'.format(i)],
                                scale=P.planet['rp_err{}'.format(i)],
                                size=len(chains)
                            )

                            _set_param('rp', rp)
                            _set_param('rhop', radvel.utils.density(mpsini, rp))

                            outcols.append(_get_colname('rhop'))
                        except (AttributeError, KeyError):
                            pass
                    """

                    ########
                    savestate = {'run': True}
                    outcols = [par for par in synthchains.columns if par.startswith("a") 
                                                                  or "mpsini" in par 
                                                                  or "musini" in par]
                    ########


                    #import pdb; pdb.set_trace()
                    # Get quantiles and update posterior object to median
                    # values returned by MCMC chains
                    #quantiles = chains.quantile([0.159, 0.5, 0.841])
                    quantiles = synthchains.quantile([0.159, 0.5, 0.841])
                    csvfn = os.path.join(outdir, self.setup_name+'_derived_quantiles.csv')
                    quantiles.to_csv(csvfn, columns=outcols)


                    #import pdb; pdb.set_trace()
                    # save derived parameters to posterior file
                    postfile = os.path.join(outdir,
                            'post_final.pkl')
                    self.post.derived = quantiles[outcols]
                    self.post.writeto(postfile)
                    savestate['quantfile'] = os.path.relpath(csvfn)

                    csvfn = os.path.join(outdir, self.setup_name+'_derived.csv.bz2')
                    #import pdb; pdb.set_trace()
                    synthchains.to_csv(csvfn, columns=outcols, compression='bz2')
                    savestate['chainfile'] = os.path.relpath(csvfn)

                    #print("Derived parameters:", outcols)

                    driver.save_status(statfile, 'derive', savestate)
                    #import pdb; pdb.set_trace()
                    ##################################################################

                    ## Load statfile to update status
                    status = driver.load_status(statfile)



                    #############################################################
                    # Make corner and multipanel plots. Do this AFTER getting derived params
                    labels = []

                    for key_label in logpost.list_vary_params(): # Just add labels for the params that vary
                        labels.append(key_label)
                    texlabels = [self.post.params.tex_labels().get(l, l)
                                 for l in labels]

                    if len(labels)>0:
                        # labels will be empty if nothing is found --> all params fixed
                        
                        try:
                            
                            plot = corner.corner(synthchains[labels], labels=texlabels,
                                             label_kwargs={"fontsize": 14},
                                             plot_datapoints=False, bins=30,
                                             quantiles=[0.16, 0.5, 0.84],
                                             title_kwargs={"fontsize": 14},
                                             show_titles=True, smooth=True)
                            corner_save_dir = outdir+'/{}_corner_plot.pdf'.format(self.starname)
                            pl.savefig(corner_save_dir)
                            corner_plot=True
                        except ValueError as err:
                            #import pdb; pdb.set_trace()
                            print("Could not produce corner plot. Continuing.")


                    # Generate an orbit plot wth median parameters and uncertainties.
                    multipanel_save_dir = outdir+"/{}_rv_multipanel.pdf".format(self.setup_name)
                    rvplot = orbit_plots.MultipanelPlot(self.post,saveplot=
                                                        multipanel_save_dir, uparams=self.post.uparams,
                                                        status=status)
                    #import pdb; pdb.set_trace()
                    multiplot_fig, ax_list = rvplot.plot_multipanel()
                    
                    multipanel_plot=True
                    ############################################################
                    #import pdb; pdb.set_trace()



                    if 'derive' in status.sections() and status.getboolean('derive', 'run'):
                        dchains = pd.read_csv(status.get('derive', 'chainfile'))
                        synthchains = synthchains.join(dchains, rsuffix='_derived')
                        derived = True
                    else:
                        derived = False
                    try:
                        compstats = eval(status.get('ic_compare', 'ic'))

                    except Exception as err: 
                        print("WARNING: No model comparison performed.")
                        compstats = None

                    ## Judah: not sure why, but I have to delete some (any?) columns to avoid an error in compiling the report. Removing hte _derived columns removes no vital info from the reports.
                    delete_cols = [par for par in synthchains if "_derived" in par]
                    synthchains = synthchains.drop(columns=delete_cols)

                    report = radvel.report.RadvelReport(P, self.post, synthchains, minafactor, maxarchange, maxgr, mintz, compstats=compstats, derived=derived)
                    report.runname = self.setup_name
                    print("CORNER AND MULTI", corner_plot, multipanel_plot)

                    if corner_plot:
                        plot_savestate = {'corner_plot':os.path.relpath(corner_save_dir)}
                        driver.save_status(statfile, 'plot', plot_savestate)
                        status = driver.load_status(statfile)

                    if multipanel_plot:
                        plot_savestate = {'multipanel_plot':os.path.relpath(multipanel_save_dir)}
                        driver.save_status(statfile, 'plot', plot_savestate)
                        status = driver.load_status(statfile)


                    report_depfiles = []
                    for ptype, pfile in status.items('plot'):
                        report_depfiles.append(pfile)

                    with radvel.utils.working_directory(outdir):
                        rfile = os.path.join(self.setup_name+"_results.pdf")
                        report_depfiles = [os.path.basename(p) for p in report_depfiles]
                        #import pdb; pdb.set_trace()
                        report.compile(
                            rfile, depfiles=report_depfiles, latex_compiler="pdflatex"
                        )


            self.save(filename=post_path) # Write post to this file

            pickle_out = open(outdir+'/search.pkl','wb')
            pickle.dump(self, pickle_out)
            pickle_out.close()
            #import pdb; pdb.set_trace()


            periodograms_plus_pers = np.append([self.pers], list(self.periodograms.values()), axis=0).T
            np.savetxt(outdir+'/pers_periodograms.csv', periodograms_plus_pers,
                       header='period  BIC_array')

            if fixed_threshold is None:
                threshs_bics_faps = np.append([list(self.bic_threshes.values())],
                                              [list(self.best_bics.values()),
                                               list(self.eFAPs.values())], axis=0).T

                np.savetxt(outdir+'/thresholds_bics_faps.csv', threshs_bics_faps,
                           header='threshold  best_bic  fap')

            

    def continue_search(self, fixed_threshold=True, running=True):
        """Continue a search by trying to add one more planet

        Args:
            fixed_threshold (bool): fix the BIC threshold at the last threshold, or re-derive for each periodogram
        """
        if self.num_planets == 0:
            self.add_planet()

        last_thresh = max(self.bic_threshes.keys())

        if fixed_threshold:
            #thresh = self.bic_threshes[last_thresh]
            #print(self.starname, thresh)
            thresh = 30 # Judah addition to test specific systems
        else:
            thresh = None

        #print("THRESH in continue_search", thresh)

        ## Judah addition. Move to specified basis to fix params. Stay in that basis for run_search()
        self.post.params.basis.to_any_basis(self.post.params, "per tc secosw sesinw k")

        # Fix parameters of all known planets.
        if self.num_planets != 0:
            for n in np.arange(self.num_planets):
                self.post.params['per{}'.format(n+1)].vary    = False
                self.post.params['tc{}'.format(n+1)].vary     = False
                self.post.params['k{}'.format(n+1)].vary      = False
                self.post.params['secosw{}'.format(n+1)].vary = False
                self.post.params['sesinw{}'.format(n+1)].vary = False
     
                ## Manually added by Judah: params in likelihood should be fixed too.
                self.post.likelihood.params['per{}'.format(n+1)].vary    = False
                self.post.likelihood.params['tc{}'.format(n+1)].vary     = False
                self.post.likelihood.params['k{}'.format(n+1)].vary      = False
                self.post.likelihood.params['secosw{}'.format(n+1)].vary = False
                self.post.likelihood.params['sesinw{}'.format(n+1)].vary = False
        

        #print("Params in continue_search", *[self.post.params["per{}".format(p+1)] for p in range(self.post.params.num_planets)], sep="\n")
        self.run_search(fixed_threshold=thresh, mkoutdir=False, running=running)
        #print("BIC at the end of continue_search", self.post.bic())

    def inject_recover(self, injected_orbel, num_cpus=None, full_grid=False):
        """Inject and recover
        Inject and attempt to recover a synthetic planet signal
        Args:
            injected_orbel (array): array of orbital elements sent to radvel.kepler.rv_drive
            num_cpus (int): Number of CPUs to utilize. Will default to self.workers
            full_grid (bool): if True calculate periodogram on full grid, if False only calculate
                at single period
        Returns:
            tuple: (recovered? (T/F), recovered_orbel)
        """

        if num_cpus is not None:
            self.workers = int(num_cpus)


        self.max_planets = self.num_planets + 1
        #print("THE MAX planets for {} is {}".format(self.starname, self.max_planets))
        self.mcmc = False
        self.save_outputs = False
        self.verbose = False
        #import pdb; pdb.set_trace()
        # 8/2: Trying to fix injections, possibly basebic error.
        self.basebic = None
        if not full_grid:
            self.manual_grid = [injected_orbel[0]] # Judah comment: isn't this cheating? We should be blind to injected period.
            fixed_threshold = True
        else:
            #print("full_grid worked")
            fixed_threshold = True

            # These are the pers to search over, not to inject over (those are in the inject object)
            min_per=np.min(self.pers)
            max_per=np.max(self.pers)
            man_grid_len = int(max(len(self.pers)/4, 50))
            # Equal frequency spacing:
            self.manual_grid = 1/np.linspace(1/min_per, 1/max_per, man_grid_len)
            


        ## Judah addition: make sure we're in synthesis basis before generating model RVs.
        self.post.params.basis.to_any_basis(self.post.params, "per tp e w k")
        mod = radvel.kepler.rv_drive(self.data['time'].values, injected_orbel)
        

        pltt = False
        if pltt:
            import matplotlib.pyplot as plt
            plt.close()
            #plt.scatter(self.data.time, self.data.mnvel, c='r', label='data.mnvel')
            #plt.scatter(self.post.likelihood.x, self.post.likelihood.y, facecolor="None", edgecolor='b', label='likelihood.y')

            data_x = self.post.likelihood.x
            data_y = self.post.likelihood.y

            model_x = np.linspace(np.min(data_x), np.max(data_x), 400)
            model_y = self.post.model(model_x)#radvel.kepler.rv_drive(model_x, injected_orbel)

            plt.scatter(data_x, data_y, c='b', label='after')
            plt.plot(model_x, model_y, c='b', label='injected model')

            plt.legend()
            #plt.ylim(-20, 20)
            #plt.savefig("data_check_I_guess.png", dpi=300)
            plt.savefig("before_injections.png")
            plt.close()


        self.data['mnvel'] += mod

        self.post = utils.initialize_post(self.data, self.post.params, self.post.priors, time_base=self.time_base) # Judah addition to include updated data in post. Might be redundant with add_planet() lines, but seems to make a difference.

        if pltt:

            data_x = self.post.likelihood.x
            data_y = self.post.likelihood.y


            model_x = np.linspace(np.min(data_x), np.max(data_x), 400)
            model_y = self.post.model(model_x)

            plt.scatter(data_x, data_y, c='b', label='after')
            plt.plot(model_x, model_y, c='b', label='injected model')
            plt.legend()
            plt.savefig("after_injection.png")
            plt.close()
        
        self.continue_search(fixed_threshold, running=False)

        # Determine successful recovery
        last_planet = self.num_planets
        pl = str(last_planet)
        #import pdb; pdb.set_trace()
        if last_planet >= self.max_planets:
            synth_params = self.post.params.basis.to_synth(self.post.params)
            recovered_orbel = [synth_params['per'+pl].value,
                               synth_params['tp'+pl].value,
                               synth_params['e'+pl].value,
                               synth_params['w'+pl].value,
                               synth_params['k'+pl].value]
            per, tp, e, w, k = recovered_orbel
            iper, itp, ie, iw, ik = injected_orbel

            # calculate output model to check for phase mismatch
            # probably not most efficient way to do this
            xmod = np.linspace(tp, tp+iper, 100)
            inmod = radvel.kepler.rv_drive(xmod, injected_orbel)
            outmod = self.post.likelihood.model(xmod)
            xph1 = np.mod(xmod - itp, iper)
            xph1 /= iper
            xph2 = np.mod(xmod - tp, per)
            xph2 /= per
            inmin = xph1[np.argmin(inmod)]
            outmin = xph2[np.argmin(outmod)]
            inmax = xph1[np.argmax(inmod)]
            outmax = xph2[np.argmax(outmod)]
            phdiff = np.min([abs(inmin - outmin), abs(outmax - inmax)])

            dthresh = 0.25                                 # recover parameters to 25%
            criteria = [last_planet >= self.max_planets,   # check detected
                        np.abs(per-iper)/iper <= dthresh,  # check periods match
                        phdiff <= np.pi / 6,               # check that phase is right
                        np.abs(np.abs(k) - ik)/ik <= dthresh]      # check that K is right
            #import pdb; pdb.set_trace()
            criteria = np.array(criteria, dtype=bool)
            #print("Critical", criteria)
            #if np.array([criteria[0], criteria[1], criteria[3]]).all() and (not criteria[2]):
                #print("Just the phase")
            

            if criteria.all():
                recovered = True
            else:
                recovered = False

        else:
            recovered = False
            recovered_orbel = [np.nan for i in range(5)]
        
        # Separately check if system has recovered trend
        if self.trend:
            self.trend_pref = False # If I set to False here, do I need to set in in trend_test()?
            dvdt = self.post.params['dvdt'].value
            curv = self.post.params['curv'].value

            trend_floor = 8/(3*365.25) # 8 m/s RV variation over 3 years is a 4sigma detection (~2m/s errors)
            curv_floor = 8/(3*365.25)**2
   
            bic_condition = self.trend_bic_diff < -30 # Did trend win over flat by 30?

            trend_curv_condition = (abs(dvdt)>trend_floor or abs(curv)>curv_floor)
            if bic_condition and trend_curv_condition:
                trend_pref = True
            else:
                trend_pref = False
                
            trendel = [self.post.params['dvdt'].value, self.post.params['curv'].value]
        else:
            trend_pref = False
            trendel = [np.nan for i in range(2)]
        #print("TREND PREF", trend_pref)
        plot_final = False
        if plot_final:

            ## First, set all planet signals to K=0 except the one we're plotting
            #for i in range(self.post.params.num_planets-1):
             #   k_key = "k{}".format(i+1)
              #  self.post.params[k_key].value=0

            import matplotlib.pyplot as plt
            plt.close()

            #colors = ['red', 'green', 'purple', 'orange']
            #for i, tel_name in enumerate(self.tels):
                #data_tel = self.data.query('tel=="{}"'.format(tel_name))
                #inds = np.where(self.post.likelihood.telvec==tel_name)

                #data_x = self.post.likelihood.x[inds] #data_tel.time
                #data_y = self.post.likelihood.y[inds] #data_tel.mnvel
                #color = colors[i]
                #offset = self.post.params['gamma_{}'.format(tel_name)].value
                #plt.scatter(data_x, data_y, c=color, label='data_{}'.format(tel_name))
            

            data_x = self.post.likelihood.x
            data_y = self.post.likelihood.y
            #plt.scatter(data_x, data_y, c='red', label='data_{}'.format(tel_name))
            fig, ax = plt.subplots()
            ldata = self.data
            radvel.mtelplot(ldata.time, ldata.mnvel, ldata.errvel, ldata.tel, ax)

            model_x = np.linspace(np.min(data_x), np.max(data_x)+300, 200)
            model_y = self.post.likelihood.model(model_x) + self.post.params['gamma_j'].value

            inj_x = np.linspace(np.min(data_x), np.max(data_x), 200)
            inj_y = radvel.kepler.rv_drive(inj_x, injected_orbel)#+self.post.params['gamma_j'].value


            plt.plot(inj_x, inj_y, c='black', label='inj')
            plt.plot(model_x, model_y, c='blue', label='model')


            plt.legend()
            plt.savefig("final_model.png")
            plt.close()
            #import pdb; pdb.set_trace()

        return recovered, recovered_orbel, trend_pref, trendel



