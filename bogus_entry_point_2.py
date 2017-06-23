import importlib.util
import sys,time
import os
import argparse
import threading
import logging
from mpi4py import MPI
import inspect

logging.basicConfig(level=logging.INFO)
# comm = MPI.COMM_WORLD

# Add directory containing this script to the python path
# thisFile = inspect.stack()[0][1]
# logging.info("Running " + thisFile)
# base = os.path.dirname(os.path.abspath(thisFile))
# logging.info("Adding " + base + " to python path")
# sys.path.append(os.path.join(base))

class BogusProgressReporter(object):
    def __init__(self, period, max_it=None):
        self.period = period
        self.it = 0
        self.max_it = max_it
        self.timer = None

    def start(self):
        print("Inside Start")
        self.timer = threading.Timer(self.period, self.report)
        self.timer.start()

    def should_stop(self):
        print("Inside Stop")
        return (self.timer is None) or (self.max_it and self.it > self.max_it)

    def report(self):
        print(self.it)
        self.it += 1
        if not self.should_stop():
            logging.info("\nPROGRESS: {:.2%}".format(1.0 - 1.0 / self.it))
            self.start()
        else:
            self.stop()

    def stop(self):
        if self.timer:
            self.timer.cancel()
            self.timer = None
        logging.info("PROGRESS: 100%")
        sys.stdout.flush()
return_code = 1

if __name__ == '__main__':

    # if comm.rank != 0:
    #     sys.exit(0)

    parser = argparse.ArgumentParser(description='Philly wrapper.')
    # parser.add_argument('-datadir', type=str, nargs='?', help='path of the data directory')
    # parser.add_argument('-logdir', type=str, nargs='?', help='path of the log directory')
    # parser.add_argument('-outputdir', type=str, help='path of the model directory')

    parser.add_argument('-bogus_progress_period', type=int, default=10,
                        help='print bogus PROGRESS: pct% message every N seconds.')
    args, unknown = parser.parse_known_args()

    # old_argv = list(sys.argv)
    # sys.argv.pop(0)

    # script = None
    # k = sys.argv.index('--run_this')
    # _ = sys.argv.pop(k)
    # script = sys.argv.pop(k)

    # import shutil
    # #shutil.copytree(args.outputdir, args.logdir)

    # sys.argv += ["--philly_job", "True"]
    # if args.datadir is not None:
    #     sys.argv += ["--data_dir", args.datadir]
    # if args.outputdir is not None:
    #     sys.argv += ["--model_dir", args.outputdir]

    # script_path = os.path.join(base, script)

    # logging.info("Launching " + script_path)
    # sys.argv.insert(0, script_path)

    # logging.info("Args: {}".format(sys.argv))

    # spec = importlib.util.spec_from_file_location("__main__", script_path)
    # logging.info(spec)

    # module_to_run = importlib.util.module_from_spec(spec)
    # sys.modules['__main__'] = module_to_run

    print(args.bogus_progress_period)
    bogus_progress = None
    try:
        if args.bogus_progress_period != 0:
            bogus_progress = BogusProgressReporter(args.bogus_progress_period)
            bogus_progress.start()
        for i in range(30):
            time.sleep(2)
            if i%5==0:
                print(i)
        # spec.loader.exec_module(module_to_run)
        return_code = 0
    except SystemExit as e:
        if e.code:
            return_code = e.code
        else:
            return_code = 0
    except:
        import traceback
        logging.error(traceback.print_exc())
    finally:
        if bogus_progress:
            print("Before Finally Stop")
            bogus_progress.stop()

sys.exit(return_code)
