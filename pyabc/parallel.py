import logging
import multiprocessing as mp


class Parallel(object):

    def __init__(self, progressbar, processes=mp.cpu_count()):
        self.proc = processes
        self.results = None
        self.bar = progressbar
        # logging.info('\n')
        logging.info('Parallel regime: {} workers'.format(processes))
        logging.info('Available {} workers'.format(mp.cpu_count()))

    def run(self, func, tasks):
        pool = mp.Pool(processes=self.proc)
        if self.bar == 1:
            logging.debug('Progress bar changed')
            try:
                from tqdm import tqdm
            except ImportError:
                self.bar = 0

        if self.bar == 1:
            logging.debug('Progress bar with tqdm package')
            self.results = []
            with tqdm(total=len(tasks)) as pbar:
                for i, res in tqdm(enumerate(pool.imap_unordered(func, tasks)), desc='ABC algorithm'):
                    self.results.append(res)
                    pbar.update()
            pbar.close()
        elif self.bar == 2:
            logging.debug('Printing progress')
            self.results = pool.map_async(func, tasks)
            while not self.results.ready():
                done = len(tasks) - self.results._number_left * self.results._chunksize
                logging.info("Done {}% ({}/{})".format(int(done/len(tasks)*100), done, len(tasks)))
            pool.close()
            pool.join()
        else:
            logging.debug('No progress bar')
            self.results = pool.map(func, tasks)
            pool.close()
        pool.terminate()

    def get_results(self):
        if self.bar == 1:
            return [x for x in self.results]
        if self.bar == 2:
            return self.results.get()
        else:
            return self.results

