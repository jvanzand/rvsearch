"""
Command Line Interface
"""
import os
from argparse import ArgumentParser
import warnings

import rvsearch
import rvsearch.driver
import rvsearch.periodogram

warnings.simplefilter("ignore")
warnings.simplefilter('once', DeprecationWarning)


def main():
    psr = ArgumentParser(
        description="RadVel-Search: Automated planet detection pipeline", prog='rvsearch'
    )
    psr.add_argument('--version',
        action='version',
        version="%(prog)s {}".format(rvsearch.__version__),
        help="Print version number and exit."
    )

    subpsr = psr.add_subparsers(title="subcommands", dest='subcommand')

    # In the parent parser, we define arguments and options common to
    # all subcommands.
    psr_parent = ArgumentParser(add_help=False)
    psr_parent.add_argument('--num_cpus',
                            action='store', default=8, type=int,
                            help="Number of CPUs [8]")
    psr_parent.add_argument('-d', '--search_dir', metavar='search directory',
                          type=str,
                          help="path to existing radvel-search output directory"
                          )
    psr_parent.add_argument('--verbose',
                          action='store_true', default=True,
                          help="Print extra messages and progress bars"
                          )
    psr_parent.add_argument('-o', '--output_dir', metavar='output directory',
                          type=str,
                          help="path to place outputs", default=None
                          )


    # Search
    psr_search = subpsr.add_parser('find', parents=[psr_parent], )
    psr_search.add_argument('-s', '--setupfn', metavar="RadVel setup file", type=str,
                            help="Path to RadVel setup file.")
    psr_search.add_argument('--minP',
                          type=float, action='store', default=1.2,
                          help="Minimum search period [default=1.2]"
                          )
    # psr_search.add_argument('--maxP',
    #                       type=float, action='store', default=1e4,
    #                       help="Maximum search period [default=10000]"
    #                       )
    psr_search.add_argument('--known', action='store_true',
                          help="Include previously known planets [default=False]"
                          )
    psr_search.add_argument('--num_freqs',
                          action='store', default=None,
                          help="Number of test frequencies"
                          )
    psr_search.add_argument('--trend',
                          action='store_true',
                          help="Trend free during periodogram calculation [default=False]",
                          )
    psr_search.add_argument('--mcmc', action='store_true',
                          help="Run MCMC after search [default=False]"
                          )
    psr_search.add_argument('--mstar',
                         type=float, action='store', default=None, nargs=2,
                         help="Stellar mass and uncertainty [msun]. Will use values defined in setup file if not provided on the command line."
                         )
    psr_search.add_argument('--maxplanets',
                         type=int, action='store', default=8,
                         help="Maximum number of planets to search for (including pre-defined planets) [default=8]"
                         )


    # Injections
    psr_inj = subpsr.add_parser('inject', parents=[psr_parent], )
    psr_inj.add_argument('--minP',
                          type=float, action='store', default=3.1,
                          help="Minimum injection period [default=1.2]"
                          )
    psr_inj.add_argument('--maxP',
                          type=float, action='store', default=1e6,
                          help="Maximum injection period [default=1e6]"
                          )
    psr_inj.add_argument('--minK',
                          type=float, action='store', default=0.1,
                          help="Minimum injection K [default=0.0]"
                          )
    psr_inj.add_argument('--maxK',
                          type=float, action='store', default=1000.0,
                          help="Maximum injection K [default=1000.0]"
                          )
    psr_inj.add_argument('--minE',
                          type=float, action='store', default=0.0,
                          help="Minimum injection eccentricity [default=0.0]"
                          )
    psr_inj.add_argument('--maxE',
                          type=float, action='store', default=0.9,
                          help="Maximum injection eccentricity [default=0.9]"
                          )
    psr_inj.add_argument('--rstar',
                          type=float, action='store', default=None,
                          help="Stellar radius [default=None]"
                          )
    psr_inj.add_argument('--teff',
                          type=float, action='store', default=None,
                          help="Stellar effective temperature [default=None]"
                          )

    psr_inj.add_argument('--num_inject',
                          type=int, action='store', default=3000,
                          help="Number of injections [default=3000]"
                          )
    psr_inj.add_argument('--full_grid',
                          action='store_true',
                          help="Run search over full period grid [default=False]"
                          )
    psr_inj.add_argument('--overwrite',
                          action='store_true',
                          help="Force overwrite [default=False]"
                          )
    psr_inj.add_argument('--betaE',
                         action='store_true',
                         help="Inject using the Kipping 2013 beta distribution for ecc [default=False]"
                         ) 


    psr_search.set_defaults(func=rvsearch.driver.run_search)
    psr_inj.set_defaults(func=rvsearch.driver.injections)

    # Plots
    psr_plot = subpsr.add_parser('plot', parents=[psr_parent],)
    psr_plot.add_argument('-t', '--type',
                          type=str, nargs='+',
                          choices=['recovery', 'summary'],
                          help="type of plot(s) to generate"
                          )
    psr_plot.add_argument('--fmt',
                         type=str, action='store', default='pdf',
                         help="format to save plot [pdf]"
                         )
    psr_plot.add_argument('--trends_count',
                         type=bool, action='store_true',
                         help="Whether to count trends as recoveries"
                         )


    psr_plot.set_defaults(func=rvsearch.driver.plots)


    args = psr.parse_args()

    args.func(args)

if __name__ == '__main__':
    main()
