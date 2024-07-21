"""
Distributions
"""
from functools import wraps
from abc import ABCMeta
from abc import abstractmethod

import scipy as sp
import numpy as np

from pygam.core import Core
from pygam.utils import ylogydu


def multiply_weights(deviance):
    @wraps(deviance)
    def multiplied(self, y, mu, weights=None, **kwargs):
        if weights is None:
            weights = np.ones_like(mu)
        return deviance(self, y, mu, **kwargs) * weights

    return multiplied


def divide_weights(V):
    @wraps(V)
    def divided(self, mu, weights=None, **kwargs):
        if weights is None:
            weights = np.ones_like(mu)
        return V(self, mu, **kwargs) / weights

    return divided


class Distribution(Core):
    __metaclass__ = ABCMeta
    """
    base distribution class
    """

    def __init__(self, name=None, scale=None):
        """
        creates an instance of the Distribution class

        Parameters
        ----------
        name : str, default: None
        scale : float or None, default: None
            scale/standard deviation of the distribution

        Returns
        -------
        self
        """
        self.scale = scale
        self._known_scale = self.scale is not None
        super(Distribution, self).__init__(name=name)
        if not self._known_scale:
            self._exclude += ['scale']

    def phi(self, y, mu, edof, weights):
        """
        GLM scale parameter.
        for Binomial and Poisson families this is unity
        for Normal family this is variance

        Parameters
        ----------
        y : array-like of length n
            target values
        mu : array-like of length n
            expected values
        edof : float
            estimated degrees of freedom
        weights : array-like shape (n,) or None, default: None
            sample weights
            if None, defaults to array of ones

        Returns
        -------
        scale : estimated model scale
        """
        if self._known_scale:
            return self.scale
        else:
            return np.sum(weights * self.V(mu) ** -1 * (y - mu) ** 2) / (len(mu) - edof)

    @abstractmethod
    def sample(self, mu):
        """
        Return random samples from this distribution.

        Parameters
        ----------
        mu : array-like of shape n_samples or shape (n_simulations, n_samples)
            expected values

        Returns
        -------
        random_samples : np.array of same shape as mu
        """
        pass


class NormalDist(Distribution):
    """
    Normal Distribution
    """

    def __init__(self, scale=None):
        """
        creates an instance of the NormalDist class

        Parameters
        ----------
        scale : float or None, default: None
            scale/standard deviation of the distribution

        Returns
        -------
        self
        """
        super(NormalDist, self).__init__(name='normal', scale=scale)

    def log_pdf(self, y, mu, weights=None):
        """
        computes the log of the pdf or pmf of the values under the current distribution

        Parameters
        ----------
        y : array-like of length n
            target values
        mu : array-like of length n
            expected values
        weights : array-like shape (n,) or None, default: None
            sample weights
            if None, defaults to array of ones

        Returns
        -------
        pdf/pmf : np.array of length n
        """
        if weights is None:
            weights = np.ones_like(mu)
        scale = self.scale / weights
        return sp.stats.norm.logpdf(y, loc=mu, scale=scale)

    @divide_weights
    def V(self, mu):
        """
        glm Variance function.

        if
            Y ~ ExpFam(theta, scale=phi)
        such that
            E[Y] = mu = b'(theta)
        and
            Var[Y] = b''(theta) * phi / w

        then we seek V(mu) such that we can represent Var[y] as a fn of mu:
            Var[Y] = V(mu) * phi

        ie
            V(mu) = b''(theta) / w

        Parameters
        ----------
        mu : array-like of length n
            expected values

        Returns
        -------
        V(mu) : np.array of length n
        """
        return np.ones_like(mu)

    @multiply_weights
    def deviance(self, y, mu, scaled=True):
        """
        model deviance

        for a gaussian linear model, this is equal to the SSE

        Parameters
        ----------
        y : array-like of length n
            target values
        mu : array-like of length n
            expected values
        scaled : boolean, default: True
            whether to divide the deviance by the distribution scaled

        Returns
        -------
        deviances : np.array of length n
        """
        dev = (y - mu) ** 2
        if scaled:
            dev /= self.scale
        return dev

    def sample(self, mu):
        """
        Return random samples from this Normal distribution.

        Samples are drawn independently from univariate normal distributions
        with means given by the values in `mu` and with standard deviations
        equal to the `scale` attribute if it exists otherwise 1.0.

        Parameters
        ----------
        mu : array-like of shape n_samples or shape (n_simulations, n_samples)
            expected values

        Returns
        -------
        random_samples : np.array of same shape as mu
        """
        standard_deviation = self.scale**0.5 if self.scale else 1.0
        return np.random.normal(loc=mu, scale=standard_deviation, size=None)


class BinomialDist(Distribution):
    """
    Binomial Distribution
    """

    def __init__(self, levels=1):
        """
        creates an instance of the Binomial class

        Parameters
        ----------
        levels : int of None, default: 1
            number of trials in the binomial distribution

        Returns
        -------
        self
        """
        if levels is None:
            levels = 1
        self.levels = levels
        super(BinomialDist, self).__init__(name='binomial', scale=1.0)
        self._exclude.append('scale')

    def log_pdf(self, y, mu, weights=None):
        """
        computes the log of the pdf or pmf of the values under the current distribution

        Parameters
        ----------
        y : array-like of length n
            target values
        mu : array-like of length n
            expected values
        weights : array-like shape (n,) or None, default: None
            sample weights
            if None, defaults to array of ones

        Returns
        -------
        pdf/pmf : np.array of length n
        """
        if weights is None:
            weights = np.ones_like(mu)
        n = self.levels
        p = mu / self.levels
        return sp.stats.binom.logpmf(y, n, p)

    @divide_weights
    def V(self, mu):
        """
        glm Variance function

        computes the variance of the distribution

        Parameters
        ----------
        mu : array-like of length n
            expected values

        Returns
        -------
        variance : np.array of length n
        """
        return mu * (1 - mu / self.levels)

    @multiply_weights
    def deviance(self, y, mu, scaled=True):
        """
        model deviance

        for a bernoulli logistic model, this is equal to the twice the
        negative loglikelihod.

        Parameters
        ----------
        y : array-like of length n
            target values
        mu : array-like of length n
            expected values
        scaled : boolean, default: True
            whether to divide the deviance by the distribution scaled

        Returns
        -------
        deviances : np.array of length n
        """
        dev = 2 * (ylogydu(y, mu) + ylogydu(self.levels - y, self.levels - mu))
        if scaled:
            dev /= self.scale
        return dev

    def sample(self, mu):
        """
        Return random samples from this Normal distribution.

        Parameters
        ----------
        mu : array-like of shape n_samples or shape (n_simulations, n_samples)
            expected values

        Returns
        -------
        random_samples : np.array of same shape as mu
        """
        number_of_trials = self.levels
        success_probability = mu / number_of_trials
        return np.random.binomial(n=number_of_trials, p=success_probability, size=None)


class PoissonDist(Distribution):
    """
    Poisson Distribution
    """

    def __init__(self):
        """
        creates an instance of the PoissonDist class

        Parameters
        ----------
        None

        Returns
        -------
        self
        """
        super(PoissonDist, self).__init__(name='poisson', scale=1.0)
        self._exclude.append('scale')

    def log_pdf(self, y, mu, weights=None):
        """
        computes the log of the pdf or pmf of the values under the current distribution

        Parameters
        ----------
        y : array-like of length n
            target values
        mu : array-like of length n
            expected values
        weights : array-like shape (n,) or None, default: None
            containing sample weights
            if None, defaults to array of ones

        Returns
        -------
        pdf/pmf : np.array of length n
        """
        if weights is None:
            weights = np.ones_like(mu)
        # in Poisson regression weights are proportional to the exposure
        # so we want to pump up all our predictions
        # NOTE: we assume the targets are counts, not rate.
        # ie if observations were scaled to account for exposure, they have
        # been rescaled before calling this function.
        # since some samples have higher exposure,
        # they also need to have higher variance,
        # we do this by multiplying mu by the weight=exposure
        mu = mu * weights
        return sp.stats.poisson.logpmf(y, mu=mu)

    @divide_weights
    def V(self, mu):
        """
        glm Variance function

        computes the variance of the distribution

        Parameters
        ----------
        mu : array-like of length n
            expected values

        Returns
        -------
        variance : np.array of length n
        """
        return mu

    @multiply_weights
    def deviance(self, y, mu, scaled=True):
        """
        model deviance

        for a bernoulli logistic model, this is equal to the twice the
        negative loglikelihod.

        Parameters
        ----------
        y : array-like of length n
            target values
        mu : array-like of length n
            expected values
        scaled : boolean, default: True
            whether to divide the deviance by the distribution scaled

        Returns
        -------
        deviances : np.array of length n
        """
        dev = 2 * (ylogydu(y, mu) - (y - mu))

        if scaled:
            dev /= self.scale
        return dev

    def sample(self, mu):
        """
        Return random samples from this Poisson distribution.

        Parameters
        ----------
        mu : array-like of shape n_samples or shape (n_simulations, n_samples)
            expected values

        Returns
        -------
        random_samples : np.array of same shape as mu
        """
        return np.random.poisson(lam=mu, size=None)


class GammaDist(Distribution):
    """
    Gamma Distribution
    """

    def __init__(self, scale=None):
        """
        creates an instance of the GammaDist class

        Parameters
        ----------
        scale : float or None, default: None
            scale/standard deviation of the distribution

        Returns
        -------
        self
        """
        super(GammaDist, self).__init__(name='gamma', scale=scale)

    def log_pdf(self, y, mu, weights=None):
        """
        computes the log of the pdf or pmf of the values under the current distribution

        Parameters
        ----------
        y : array-like of length n
            target values
        mu : array-like of length n
            expected values
        weights : array-like shape (n,) or None, default: None
            containing sample weights
            if None, defaults to array of ones

        Returns
        -------
        pdf/pmf : np.array of length n
        """
        if weights is None:
            weights = np.ones_like(mu)
        nu = weights / self.scale
        return sp.stats.gamma.logpdf(x=y, a=nu, scale=mu / nu)

    @divide_weights
    def V(self, mu):
        """
        glm Variance function

        computes the variance of the distribution

        Parameters
        ----------
        mu : array-like of length n
            expected values

        Returns
        -------
        variance : np.array of length n
        """
        return mu**2

    @multiply_weights
    def deviance(self, y, mu, scaled=True):
        """
        model deviance

        for a bernoulli logistic model, this is equal to the twice the
        negative loglikelihod.

        Parameters
        ----------
        y : array-like of length n
            target values
        mu : array-like of length n
            expected values
        scaled : boolean, default: True
            whether to divide the deviance by the distribution scaled

        Returns
        -------
        deviances : np.array of length n
        """
        dev = 2 * ((y - mu) / mu - np.log(y / mu))

        if scaled:
            dev /= self.scale
        return dev

    def sample(self, mu):
        """
        Return random samples from this Gamma distribution.

        Parameters
        ----------
        mu : array-like of shape n_samples or shape (n_simulations, n_samples)
            expected values

        Returns
        -------
        random_samples : np.array of same shape as mu
        """
        # in numpy.random.gamma, `shape` is the parameter sometimes denoted by
        # `k` that corresponds to `nu` in S. Wood (2006) Table 2.1
        shape = 1.0 / self.scale
        # in numpy.random.gamma, `scale` is the parameter sometimes denoted by
        # `theta` that corresponds to mu / nu in S. Wood (2006) Table 2.1
        scale = mu / shape
        return np.random.gamma(shape=shape, scale=scale, size=None)


class InvGaussDist(Distribution):
    """
    Inverse Gaussian (Wald) Distribution
    """

    def __init__(self, scale=None):
        """
        creates an instance of the InvGaussDist class

        Parameters
        ----------
        scale : float or None, default: None
            scale/standard deviation of the distribution

        Returns
        -------
        self
        """
        super(InvGaussDist, self).__init__(name='inv_gauss', scale=scale)

    def log_pdf(self, y, mu, weights=None):
        """
        computes the log of the pdf or pmf of the values under the current distribution

        Parameters
        ----------
        y : array-like of length n
            target values
        mu : array-like of length n
            expected values
        weights : array-like shape (n,) or None, default: None
            containing sample weights
            if None, defaults to array of ones

        Returns
        -------
        pdf/pmf : np.array of length n
        """
        if weights is None:
            weights = np.ones_like(mu)
        gamma = weights / self.scale
        return sp.stats.invgauss.logpdf(y, mu, scale=1.0 / gamma)

    @divide_weights
    def V(self, mu):
        """
        glm Variance function

        computes the variance of the distribution

        Parameters
        ----------
        mu : array-like of length n
            expected values

        Returns
        -------
        variance : np.array of length n
        """
        return mu**3

    @multiply_weights
    def deviance(self, y, mu, scaled=True):
        """
        model deviance

        for a bernoulli logistic model, this is equal to the twice the
        negative loglikelihod.

        Parameters
        ----------
        y : array-like of length n
            target values
        mu : array-like of length n
            expected values
        scaled : boolean, default: True
            whether to divide the deviance by the distribution scaled

        Returns
        -------
        deviances : np.array of length n
        """
        dev = ((y - mu) ** 2) / (mu**2 * y)

        if scaled:
            dev /= self.scale
        return dev

    def sample(self, mu):
        """
        Return random samples from this Inverse Gaussian (Wald) distribution.

        Parameters
        ----------
        mu : array-like of shape n_samples or shape (n_simulations, n_samples)
            expected values

        Returns
        -------
        random_samples : np.array of same shape as mu
        """
        return np.random.wald(mean=mu, scale=self.scale, size=None)



import numpy as np
import scipy as sp
from scipy import stats

class TweedieDist(Distribution):
    """
    Tweedie Distribution
    """
    def __init__(self, p=1.5):
        """
        Creates an instance of the TweedieDist class
        Parameters
        ----------
        p : float, default=1.5
            The power parameter of the Tweedie distribution
        Returns
        -------
        self
        """
        super(TweedieDist, self).__init__(name='tweedie', scale=1.0)
        self._exclude.append('scale')
        self.p = p

    def log_pdf(self, y, mu, phi, weights=None):
        """
        Computes the log of the pdf of the values under the current distribution
        Parameters
        ----------
        y : array-like of length n
            target values
        mu : array-like of length n
            expected values
        phi : float
            dispersion parameter
        weights : array-like shape (n,) or None, default: None
            containing sample weights
            if None, defaults to array of ones
        Returns
        -------
        pdf : np.array of length n
        """
        if weights is None:
            weights = np.ones_like(mu)
        
        # Tweedie log-likelihood calculation
        a = -1 / (self.p - 1)
        b = -1 / (2 - self.p)
        c = (y**(2-self.p)) / ((2-self.p) * phi)
        d = (mu**(2-self.p)) / ((2-self.p) * phi)
        e = (y * mu**(1-self.p)) / ((1-self.p) * phi)
        
        return weights * (a * np.log(y/mu) + b * np.log(phi) + c - d + e)

    @divide_weights
    def V(self, mu):
        """
        GLM Variance function
        Computes the variance of the distribution
        Parameters
        ----------
        mu : array-like of length n
            expected values
        Returns
        -------
        variance : np.array of length n
        """
        return mu**self.p

    @multiply_weights
    def deviance(self, y, mu, scaled=True):
        """
        Model deviance
        Parameters
        ----------
        y : array-like of length n
            target values
        mu : array-like of length n
            expected values
        scaled : boolean, default: True
            whether to divide the deviance by the distribution scale
        Returns
        -------
        deviances : np.array of length n
        """
        dev = 2 * ((y**(2-self.p))/(2-self.p) - y*mu**(1-self.p)/(1-self.p) + mu**(2-self.p)/(2-self.p))
        if scaled:
            dev /= self.scale
        return dev

    def sample(self, mu, phi, size=None):
        """
        Return random samples from this Tweedie distribution.
        Parameters
        ----------
        mu : array-like of shape n_samples or shape (n_simulations, n_samples)
            expected values
        phi : float
            dispersion parameter
        size : int or tuple of ints, optional
            Output shape. If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn.
        Returns
        -------
        random_samples : np.array of same shape as mu
        """
        # Note: This is a simplified sampling method and may not be accurate for all Tweedie distributions
        gamma_shape = (2 - self.p) / (self.p - 1)
        gamma_scale = phi * (mu**(self.p - 1)) / (2 - self.p)
        
        num_poisson = np.random.poisson(mu**(2-self.p) / (phi * (2-self.p)), size=size)
        gamma_samples = np.random.gamma(shape=gamma_shape, scale=gamma_scale, size=size)
        
        return np.sum(gamma_samples, axis=-1) * (num_poisson > 0)


DISTRIBUTIONS = {
    'normal': NormalDist,
    'poisson': PoissonDist,
    'binomial': BinomialDist,
    'gamma': GammaDist,
    'inv_gauss': InvGaussDist,
    'tweedie': TweedieDist,
}
