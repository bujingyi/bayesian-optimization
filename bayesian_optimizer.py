import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import *


class BayesianOptimizer:
    def __init__(
            self,
            func,
            float_param_ranges={},
            int_param_candidates={},
            n_init_points=10000,
            external_init_points=None,
            max_iter=1e4,
            no_new_converge=3,
            no_better_converge=10,
            kernel=RBF(),
            acq_type='PI',
            beta_lcb=0.5,
            eps=1e-7,
            n_sample=int(1e6),
            seed=None
    ):
        """
        Initializer with initial points and estimate the GPR landscape
        :param func: objective function to optimize
        :param float_param_ranges: float parameters' ranges. Dictionary {'param_name': range}, range is a tuple of shape
            (low, high)
        :param int_param_candidates: integer parameters' candidates. Dictionary {'param_name': candidate}, candidate is
            a list
        :param n_init_points: number of initial points to sample to fit GPR
        :param external_init_points: dictionary of param and list of values, {p1:[pc1,pc2,...], p2:[pc1,pc2,...], ...}
        :param max_iter: max number of iterations
        :param no_new_converge: number of continued iterations with no new points sampled from GPR acquisition
        :param no_better_converge: number of continued iterations with no improvement on performance evaluation
        :param kernel: a kernel for GPR
        :param acq_type: type of acquisition function
        :param beta_lcb: beta coefficient used in Lower Confidence Bound acquisition function
        :param eps: correction added to estimated standard deviation to prevent overflow in division
        :param n_sample: number of samples to generage for one iteration
        :param seed: random seed
        """
        self.func = func
        self.float_param_dict = float_param_ranges
        self.int_param_dict = int_param_candidates

        # bayesian optimization hyper-hyper-parameters
        self.max_iter = int(max_iter)
        self.no_new_converge = no_new_converge
        self.no_better_converge = no_better_converge
        self.acq_type = acq_type
        self.beta_LCB = beta_lcb
        self.eps = eps
        self.n_sample = n_sample
        self.n_init_points = n_init_points

        # random seed
        self.seed = seed

        # the underlying Gaussian Process Regression for bayesian optimization
        self.gpr = GPR(
            kernel=kernel,
            n_restarts_optimizer=50,
            random_state=self.seed
        )

        # parse hyperparameters' names
        self._parse_param_names()

        # get float parameters' ranges and integer parameters' candidates
        self._get_ranges_and_candidates()

        # get starting points
        self.init_points = self._get_init_points(external_init_points)

        self.x = self.init_points

        # initialize with points and their functional values
        print('Evaluating Initial Points...')
        self.y = np.array(
            [self.func(**self._check_int_param(dict(zip(self.param_names, p)))) for p in self.init_points]
        )

        # unique index
        u_index = self._unique_index(self.x)
        self.x = self.x[u_index]
        self.y = self.y[u_index]
        self.num_param_seeds = len(self.x)

        # initial fit for GPR
        self.gpr.fit(self.x, self.y)

    def _parse_param_names(self):
        """
        Parse parameters' names
        """
        self.float_param_names = list(self.float_param_dict.keys())
        self.int_param_names = list(self.int_param_dict.keys())
        self.param_names = self.float_param_names + self.int_param_names

    def _get_ranges_and_candidates(self):
        """
        Get float parameters' ranges and integer parameters' candidates
        """
        # float_param_ranges shape: [n_params, range]: [param1(low, high), param2(low, high) ... ]
        self.float_param_ranges = np.array(list(self.float_param_dict.values()))
        # int_param_candidates shape: [ [ candidates for param1], [ candidates for param2] ... ]
        self.int_param_candidates = list(self.int_param_dict.values())

    def _get_init_points(self, external_init_points):
        """
        Get initial points including external and internal ones for Bayesian Optimization
        :return:
        """
        internal_init_points = self._generate_random_params(self.n_init_points)

        # generate initial points
        if external_init_points is not None:
            # sanity check
            nums = np.array([len(choices) for choices in external_init_points.values()])
            if not all(nums == nums[0]):
                raise Exception('Number of values for each parameter must be the same')
            if nums.sum() != 0:
                # combine with user's input points in array, list, tuple
                points = []
                for param in self.param_names:
                    points.append(external_init_points[param])
                # points shape: [param, choices] transpose to [n point, param]
                points = np.array(points).T
                internal_init_points = np.vstack((internal_init_points, points))

        u_index = self._unique_index(internal_init_points)
        return internal_init_points[u_index]

    def _check_int_param(self, param_dict):
        """
        Convert float to integer for int parameters before passing to eval function
        :param param_dict: dictionary of parameters {'param_name': value}
        :return: 
        """
        for k, v in param_dict.items():
            if k in self.int_param_names:
                param_dict[k] = int(param_dict[k])
        return param_dict

    def _generate_random_params(self, n):
        """
        Generate random parameter combinations based on ranges and candidates
        :param n: number of samples to randomly generate
        :return: 
        """
        np.random.seed(self.seed)
        xs_range = np.random.uniform(
            low=self.float_param_ranges[:, 0],
            high=self.float_param_ranges[:, 1],
            size=(int(n), self.float_param_ranges.shape[0])
        )

        # xs_candidates shape: [num discrete param, n points]
        xs_candidates = np.array([np.random.choice(choice, size=int(n)) for choice in self.int_param_dict])
        # transpose to [n points, num parameters]
        xs_candidates = xs_candidates.T
        # xs shape: [n points, num param]
        return np.hstack((xs_range, xs_candidates))

    def _unique_index(self, xs):
        """
        Keep the indices of unique values (points)
        :param xs: 2D ndarray [num points, num params]
        :return: list of indices, integers.
        """
        uniques = np.unique(xs, axis=0)
        if len(uniques) == len(xs):
            return list(range(len(xs)))

        counter = {u: 0 for u in uniques}

        indices = []
        for i, x in enumerate(xs):
            if counter[x] == 0:
                counter[x] += 1
                indices.append(i)
        return indices

    def _acquisition_func(self, xs):
        """
        Calculate min_acquisition of a number of samples
        :param xs: 2D ndarray [num points, num params]
        :return: 1D ndarray of utility values by acquisition function. [num points]
        """
        print('Calculating utility Acquisition on sampled points based on GPR...')
        # calculate the utility of f(x) given a number of x s, regard to mean and sd
        means, sds = self.gpr.predict(xs, return_std=True)
        # Setting variance below 0 to 0
        sds[sds < 0] = 0
        # eps prevent overflow of dividing small number
        z = (self.y.min() - means) / (sds + self.eps)

        # implementation of EI, expected improvement
        if self.acq_type == 'EI':
            return (self.y.min() - means) * norm.cdf(z) + sds * norm.pdf(z)
        # implementation of PI, probability of improvement
        if self.acq_type == 'PI':
            return norm.pdf(z)
        # implementation of LCB, optimistic lower confidence bound
        if self.acq_type == 'LCB':
            return means - self.beta_LCB * sds

    def _min_acquisition(self, n=1e6):
        """
        Sample a large number of xs and evaluate their acquisition, return the best
        :param n: number of sampling points
        :return: 1D ndarray [num param]
        """
        print('Random sampling based on ranges and candidates...')
        # bounds_range shape: [num params, range]: [p1(low, high), p2(low, high) ... ]
        # xs_range shape: [n points, num numeric features]
        xs = self._generate_random_params(n)
        ys = self._acquisition_func(xs)
        return xs[ys.argmin()]

    def optimize(self):
        """
        Perform Bayesian optimization searching for best param combination
        :return: 
        """
        no_new_converge_counter = 0
        no_better_converge_counter = 0
        best = self.y.min()
        for i in range(self.max_iter):
            print('Iteration: {}, Current Best: {}'.format(i, self.y.min()))
            # check convergence
            if no_new_converge_counter > self.no_new_converge:  # no more new combination
                break
            if no_better_converge_counter > self.no_better_converge:  # no more better combination
                break

            # get one better sample from current estimated landscape
            next_best_x = self._min_acquisition(self.n_sample)

            # if x_best has been sampled before, redo sampling by going to next round
            if np.any((self.x - next_best_x).sum(axis=1) == 0):
                no_new_converge_counter += 1
                continue

            print('Iteration {}: evaluating guessed best param set by evaluation function...'.format(i))
            self.x = np.vstack((self.x, next_best_x))
            next_best_y = self.func(**self._check_int_param(dict(zip(self.param_names, next_best_x))))
            self.y = np.append(self.y, next_best_y)

            print('Iteration {}: next best is {}, {}'.format(i, next_best_y, dict(zip(self.param_names, next_best_x))))

            u_index = self._unique_index(self.x)
            self.x = self.x[u_index]
            self.y = self.y[u_index]

            # check better combination
            if self.y.min() < best:
                no_better_converge_counter = 0
                best = self.y.min()
            else:
                no_better_converge_counter += 1

            # check new combination
            if len(self.x) == self.num_param_seeds:
                no_new_converge_counter += 1
            else:
                no_new_converge_counter = 0
                self.num_param_seeds = len(self.x)

            # re-fit GPR after collecting more samples (combinations)
            print('Iteration {}: re-fit GPR with updated parameter sets'.format(i))
            self.gpr.fit(self.x, self.y)

    def get_results(self):
        """
        Get the performance evaluation on parameter sets
        :return: pd.DataFrame, [num points, num parameters + AvgTestCost + is Initial points]
        """
        num_init = len(self.init_points)
        num_new = len(self.y) - num_init
        is_init = np.array([1] * num_init + [0] * num_new).reshape((-1, 1))
        ressults = pd.DataFrame(
            np.hstack((self.x, self.y.reshape((-1, 1)), is_init)),
            columns=self.param_names+['AvgTestCost', 'isInit']
        )

        return results.sort_values(by='AvgTestCost', inplace=False)
