''' extend scipy minimize - more algorithms, better callback interface'''
from scipy.optimize import minimize as minim
import time as time
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

class StoreFun(object):
    '''function that stores the results of the last N times its been called, and prints a progress monitor'''
    def __init__(self, fun, cache_size=None, args_cache_size = 0, noisy=False, eps=None, thresh=None, maxiter=None):
        '''
        :param fun: function to wrap
        :param cache_size: how many function results to store
        :param args_cache_size: how many function arguments to store
        :param noisy: print function results
        :param eps: terminate if the difference between last application of the function and current application < eps. can be None
        :param thresh: terminate if the function value is below thresh. Can be None.
        '''
        self.fun = fun
        self.cache = []
        self.args_cache = []
        self.cache_size = cache_size
        self.args_cache_size = args_cache_size
        self.call_count = 0
        self.noisy=noisy
        self.thresh = thresh
        self.eps = eps
        self.maxiter = maxiter

    def __str__(self):
        return 'store fun'

    def __call__(self, *args, **kwargs):
        '''call the function, store the arguments and results'''
        self.last_res = self.fun(*args, **kwargs)
        self.call_count += 1

        self.cache.append(self.last_res)
        self.args_cache.append([args])

        if self.noisy:
            print(self.call_count, 'calls, result: {:.8f}'.format(self.last_res), 'min: {:.8f}'.format(np.min(self.cache)), '             ', end='\r', flush=True)

        if self.cache_size is not None:
            self.cache = self.cache[::-1][:self.cache_size][::-1]

        if self.args_cache_size is not None:
            self.args_cache = self.args_cache[::-1][:self.args_cache_size][::-1]

        if self.call_count > 1:
            if self.thresh is not None and self.last_res < self.thresh:
                raise TerminationError(args[0], self.last_res)

            if self.eps is not None and self.cache_size!=0 and np.abs(self.cache[-1]-self.cache[-2])<self.eps:
                raise TerminationError(args[0], self.last_res)

            if self.maxiter is not None and self.call_count >= self.maxiter:
                raise TerminationError(args[0], self.last_res)

        return self.last_res

class Minimize(object):
    """Minimize: Call signature same as scipy.minimize. Keeps track of history objects,
       plus result of minimization under res.res...."""
    def __init__(self, fun, x0, args=(), method=None, jac=None,
                 hess=None, hessp=None, bounds=None, constraints=(),
                 tol=None, callback=None, options={},
                 maxiter=None, progressive=False, schedule=None,
                 reverse=True, n_sweeps=1, tols=None, maxiters=None,
                 store=True, noisy=True, cache_size=None, args_cache_size=0, thresh=None, **kwargs):

        self.fun = fun if not store else StoreFun(fun, cache_size, args_cache_size, noisy=noisy, thresh=thresh, maxiter=maxiter)
        self.x0 = x0
        self.store = store
        self.noisy = noisy
        self.res = minimize(self.fun, x0, args=args, method=method, jac=jac,
                            hess=hess, hessp=hessp, bounds=bounds, constraints=constraints,
                            tol=tol, callback=callback, options=options,
                            progressive=progressive, schedule=schedule, reverse=reverse,
                            maxiter=maxiter, maxiters=maxiters,
                            n_sweeps=n_sweeps, tols=tols, **kwargs)
        if self.noisy:
            print('\n', end='')

    @property
    def last_stored_results(self):
        return np.array(self.fun.cache)

    @property
    def last_stored_arguments(self):
        return np.squeeze(np.array(self.fun.args_cache))

class MinimizeResult(object):
    def __init__(self, x, fun):
        self.x = x
        self.fun = fun

class TerminationError(Exception):
    def __init__(self, x, fun):
        self.x = x
        self.fun = fun

def central_difference(f, x, dt, i=0):
    v = np.eye(np.prod(x.shape))[i].reshape(*x.shape) # perturbation 
    d = (f(x+dt*v/2)-f(x-dt*v/2))/dt
    return d

def cgrad_f(f, dt):
    return lambda x: np.array([central_difference(f, x, dt, i) for i in range(len(x))])

def bgd(f, x, grad=None, callback=None, hyper=(1e-2, 0.9), dt=1e-3, *args, **kwargs):
    """batch gradient descent with momentum.
    :param dt: step size for finite difference gradients
    :param learning_rate: global learning rate
    :param grad: function x -> df/dt|x
    :param mass: for momentum """
    learning_rate, mass = hyper
    maxiter = kwargs['options']['maxiter'] if 'maxiter' in kwargs['options'] else 10000
    tol = kwargs['options']['gtol'] if 'gtol' in kwargs['options'] else 1e-5

    velocity = np.zeros(len(x))
    grad = cgrad_f(f, dt) if grad is None else grad
    for i in range(maxiter):
        g = grad(x)
        velocity = mass * velocity + learning_rate * grad(x)
        x = x -  velocity

        if np.linalg.norm(g) < tol:
            break
        if callback: callback(x, i, g)
    return MinimizeResult(x, f(x))

def nesterov(f, x, grad=None, callback=None, hyper=(1e-2, 0.9,), dt=1e-3, *args, **kwargs):
    """nesterov accelerated gradient"""
    learning_rate, mass = hyper
    maxiter = kwargs['options']['maxiter'] if 'maxiter' in kwargs['options'] else 10000
    tol = kwargs['options']['gtol'] if 'gtol' in kwargs['options'] else 1e-5

    velocity = np.zeros(len(x))
    grad = cgrad_f(f, dt) if grad is None else grad
    for i in range(maxiter):
        g = grad(x)
        velocity = mass * velocity + learning_rate * grad(x-mass*velocity)
        x = x -  velocity

        if np.linalg.norm(g) < tol:
            break
        if callback: callback(x, i, g)
    return MinimizeResult(x, f(x))

def RMSProp(f, x, grad=None, callback=None, hyper=(1e-2,), dt=1e-3, *args, **kwargs):
    """RMSProp"""
    learning_rate=(0.9,)
    maxiter = kwargs['options']['maxiter'] if 'maxiter' in kwargs['options'] else 10000
    tol = kwargs['options']['gtol'] if 'gtol' in kwargs['options'] else 1e-5

    velocity = np.zeros(len(x))
    grad = cgrad_f(f, dt) if grad is None else grad
    G = np.zeros_like(x)
    for i in range(maxiter):
        g = grad(x)
        G += g**2
        x = x - (learning_rate/np.sqrt(G+1e-8))*g
        if np.linalg.norm(g) < tol:
            break
        if callback: callback(x, i, g)
    return MinimizeResult(x, f(x))

def adadelta(f, x, grad=None, callback=None, hyper=(1e-2, 0.9,), dt=1e-3, *args, **kwargs):
    """adadelta"""
    learning_rate, mass = hyper
    maxiter = kwargs['options']['maxiter'] if 'maxiter' in kwargs['options'] else 10000
    tol = kwargs['options']['gtol'] if 'gtol' in kwargs['options'] else 1e-5

    velocity = np.zeros(len(x))
    grad = cgrad_f(f, dt) if grad is None else grad
    G = np.zeros_like(x)
    for i in range(maxiter):
        g = grad(x)
        G = mass*G + (1-mass)*g**2
        x = x - (learning_rate/np.sqrt(G+1e-8))*g
        if np.linalg.norm(g) < tol:
            break
        if callback: callback(x, i, g)
    return MinimizeResult(x, f(x))

def adam(f, x, grad=None, callback=None, hyper=(1e-2, 0.9, 0.999, 1e-8), dt=1e-3, *args, **kwargs):
    """adadelta"""
    learning_rate, b1, b2, e = hyper
    maxiter = kwargs['options']['maxiter'] if 'maxiter' in kwargs['options'] else 10000
    tol = kwargs['options']['gtol'] if 'gtol' in kwargs['options'] else 1e-5

    velocity = np.zeros(len(x))
    grad = cgrad_f(f, dt) if grad is None else grad
    v = np.zeros_like(x)
    m = np.zeros_like(x)
    for t in range(1, maxiter+1):
        g = grad(x)
        m = (b1*m+(1-b1)*g   )/(1-b1**t)
        v = (b2*v+(1-b2)*g**2)/(1-b2**t)
        x = x - (learning_rate/(np.sqrt(v)+e))*m
        if np.linalg.norm(g) < tol:
            break
        if callback: callback(x, i, g)
    return MinimizeResult(x, f(x))

def minim_plus(*args, **kwargs):
    '''minim plus: add new minimization routines to scipy minimize '''
    print(f'method: {kwargs["method"]} with hyperparameters {kwargs["hyper"]}')
    if kwargs['method']=='bgd':
        return bgd(*args, **kwargs)
    if kwargs['method']=='nesterov':
        return nesterov(*args, **kwargs)
    if kwargs['method']=='RMSProp':
        return RMSProp(*args, **kwargs)
    if kwargs['method']=='adadelta':
        return adadelta(*args, **kwargs)
    if kwargs['method']=='adam':
        return adam(*args, **kwargs)
    else:
        if 'hyper' in kwargs:
            del kwargs['hyper']
        return minim(*args, **kwargs)

def minimize(*args, **kwargs):
    gradient_methods = ['BFGS', 'CG', 'bgd', 'adam', 'nesterov', 'RMSProp', 'adadelta']
    (fun, x0), args = args[:2], args[2:]
    args = (fun, x0) + args
    if 'progressive' in kwargs and kwargs['progressive'] is True:
        assert kwargs['method'] in gradient_methods
        if kwargs['schedule'] is None:
            raise Exception('must provide a schedule')
        schedule = kwargs['schedule']
        reverse = kwargs['reverse']
        n_sweeps = kwargs['n_sweeps']
        del kwargs['progressive']
        del kwargs['schedule']
        del kwargs['reverse']
        del kwargs['n_sweeps']
        (fun, x0), args = args[:2], args[2:]

        gtol = 1e-5 if kwargs['tol'] is None else kwargs['tol']
        gtols = [gtol]*(len(schedule)-1) if kwargs['tols'] is None else kwargs['tols']

        maxiter = 10000 if kwargs['maxiter'] is None else kwargs['maxiter']
        maxiters = [maxiter]*(len(schedule)-1) if kwargs['maxiters'] is None else kwargs['maxiters']

        del kwargs['tols']
        del kwargs['maxiters']
        del kwargs['maxiter']

        background = x0
        call_count = 0
        for k in range(n_sweeps):
            iterator = reversed(range(len(schedule)-1)) if reverse else range(len(schedule)-1)
            print('sweep', k+1, '/', n_sweeps)
            for i in iterator:
                start, end = schedule[i:i+2]
                call_count += fun.call_count
                fun.call_count = 0

                f = lambda x: fun(np.concatenate([background[:start], x, background[end:]]))

                if kwargs['options'] is not None:
                    kwargs['options'].update({'gtol':gtols[i], 'maxiter': maxiters[i]})
                else:
                    kwargs['options'] = {'gtol':gtols[i], 'maxiter':maxiters[i]}
                print('optimizing parameters', start, 'to',  end, 'tol={}'.format(kwargs['options']['gtol']))

                kwargs.pop('tol', None)
                try:
                    res = minim_plus(f, background[start:end], tol=kwargs['options']['gtol'], *args, **kwargs)
                except TerminationError as err:
                    res = MinimizeResult(err.x, err.fun)

                background = np.concatenate([background[:start], res.x, background[end:]])
                print('')

        return MinimizeResult(background, res.fun)
    else:
        kwargs.pop('progressive', None)
        kwargs.pop('schedule', None)
        kwargs.pop('reverse', None)
        kwargs.pop('n_sweeps', None)
        kwargs.pop('tols', None)
        kwargs.pop('maxiter', None)
        kwargs.pop('maxiters', None)
        try:
            res = minim_plus(*args, **kwargs)
        except TerminationError as err:
            res = MinimizeResult(err.x, err.fun)
        return res

    def test_minimize():
        def f(x):
            return np.linalg.norm(x)**2
        maxiter, gtol, alpha = 20000, 1e-3, 1e-2
        res5 = Minimize(f, np.random.randn(15), axiter=maxiter, method='adam',  hyper=(alpha, 0, 0.3, 1), options={'gtol':gtol})
        res1 = Minimize(f, np.random.randn(15), maxiter=maxiter, method='bgd', hyper=(alpha, 0.9), options={'gtol':gtol})
        res2 = Minimize(f, np.random.randn(15), maxiter=maxiter, method='nesterov', hyper=(alpha, 0.9), options={'gtol':gtol})
        res3 = Minimize(f, np.random.randn(15), maxiter=maxiter, method='RMSProp', hyper=(alpha, ), options={'gtol':gtol})
        res4 = Minimize(f, np.random.randn(15), maxiter=maxiter, method='adadelta', hyper=(alpha, 0.9), options={'gtol':gtol})
        plt.plot(res1.last_stored_results, label='bgd')
        plt.plot(res2.last_stored_results, label='nesterov')
        plt.plot(res3.last_stored_results, label='RMSProp')
        plt.plot(res4.last_stored_results, label='adadelta')
        plt.plot(res5.last_stored_results, label='adam')
        plt.xlabel('function evaluations')
        plt.yscale('log')
        plt.legend()
        plt.savefig('compare.pdf')


if __name__=='__main__':
    test_minimize()

