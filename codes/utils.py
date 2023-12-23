'''
Define several util functions or classes for  spectral collocation method. 
'''

import numpy as np
import scipy.stats as stats
import scipy.optimize
 
from enum import Enum


###############################Some helper functions#############################
#- define functions on calcu. d1,d2,european put ptice and theta (the greek). 
#- these four values will be used later in the QD+ methods. 
def cal_d1(tau, s0, r, q, vol, strike):
    return np.log(s0 * np.exp((r-q)*tau)/strike)/(vol * np.sqrt(tau)) + 0.5*vol * np.sqrt(tau)

 
def cal_d2(tau, s0, r, q, vol, strike):
    return  cal_d1(tau, s0, r, q, vol, strike) - vol * np.sqrt(tau)

def european_option_theta(tau, s0, r, q, vol, strike):
    """put option theta"""
    r = max(r, 1e-10)
    tau = max(tau, 1e-10) # set tau negative
    d1 = cal_d1(tau, s0, r, q, vol, strike)
    d2 = cal_d2(tau, s0, r, q, vol, strike)
    return r*strike * np.exp(-r * tau) * stats.norm.cdf(-d2) - q * s0 * np.exp(-q * tau)*stats.norm.cdf(-d1) \
        - vol * s0 * np.exp(-q * tau) * stats.norm.pdf(d1)/(2 * np.sqrt(tau))

def european_put_value(tau, s0, r, q, vol, strike):
    """put option price"""
    if tau == 0:
        return max(0, strike - s0)
    d1 = cal_d1(tau, s0, r, q, vol, strike)
    d2 = cal_d2(tau, s0, r, q, vol, strike)
    return strike * np.exp(-r * tau) * stats.norm.cdf(-d2) - s0 * np.exp(-q * tau) * stats.norm.cdf(-d1)


###############################QD+ for initial guess#############################

class QDplus:
    """QD+ alogrithm for the initialized value for spectral collocation"""
    def __init__(self, r, q, sigma, K):
        self.r = r
        self.q = q
        self.sigma = sigma
        self.K = K
        self.omega = 0
        self.v_N = 0
        self.v_h = 0
        self.v_qQD = 0
        self.v_qQDdot = 0
        self.v_p = 0
        self.v_theta = 0
        self.v_c = 0
        self.v_b = 0
        self.v_d1 = 0
        self.v_d2 = 0
        self.v_dlogSdh = 0

        self.exercise_boundary = 0

        self.tolerance = 1e-10

 
    def compute_exercise_boundary(self, tau):
        if tau == 0:
            return self.K*min(1,self.r/self.q)
        res = scipy.optimize.root(self.iteration_func,x0=self.K, args=(tau,))
 
        return res.x[0]

    def iteration_func(self, S, tau):
        if tau == 0:
            return 0 if type(S) is float else np.ones(S.size) * 0
        #compute some terms to define the iteration function 
        self.omega = 2 * (self.r - self.q) / (self.sigma * self.sigma) 
        self.v_M = 2 * self.r / (self.sigma * self.sigma)
        self.v_h = 1 - np.exp(-self.r * tau) 
        self.v_qQD = -0.5*(self.omega-1) - 0.5 * np.sqrt((self.omega-1)*(self.omega-1) + 4 * self.v_M/self.v_h)
        self.v_qQDdot = self.v_M/(self.v_h * self.v_h * np.sqrt((self.omega-1)*(self.omega-1) + 4*self.v_M/self.v_h))
        self.v_d1 =  cal_d1(tau, S, self.r, self.q, self.sigma, self.K)
        self.v_d2 =  cal_d2(tau, S, self.r, self.q, self.sigma, self.K)
        self.v_p = european_put_value(tau, S, self.r, self.q, self.sigma, self.K)
        self.v_theta = european_option_theta(tau, S, self.r, self.q, self.sigma, self.K)
        self.v_dlogSdh = self.dlogSdh(tau, S)
        self.v_c0 = - (1-self.v_h)*self.v_M/(2*self.v_qQD + self.omega - 1) * (1/self.v_h - (self.v_theta*np.exp(self.r * tau))/(self.r*(self.K - S - self.v_p)) + self.v_qQDdot/(2*self.v_qQD+self.omega-1))
        self.v_c = self.v_c0 - ((1-self.v_h)*self.v_M)/(2*self.v_qQD + self.omega - 1)  * ((1 - np.exp(-self.q * tau)*stats.norm.cdf(-self.v_d1))/(self.K - S - self.v_p) + self.v_qQD/S)   * self.v_dlogSdh


        func = (1 - np.exp(-self.q * tau) * stats.norm.cdf(-self.v_d1)) * S + (self.v_qQD) * (self.K - S - self.v_p)
        return func

    def b(self, tau, S):
        N = self.omega
        M = self.v_M
        h = self.v_h
        qQD = self.v_qQD
        qQDdot = self.v_qQDdot
        p = self.v_p
        theta = self.v_theta
        c = self.v_c
        d1 = self.v_d1
        d2 = self.v_d2
        return ((1-h)*M*qQDdot)/(2*(2*qQD + N - 1))

    def dlogSdh(self, tau, S):
        N = self.omega
        M = self.v_M
        h = self.v_h
        qQD = self.v_qQD
        qQDdot = self.v_qQDdot
        p = self.v_p
        theta = self.v_theta
        c = self.v_c
        d1 = self.v_d1
        d2 = self.v_d2
        r = self.r
        q = self.q

        dFdh = qQD * theta * np.exp(self.r * tau)/self.r + qQDdot * (self.K - S - p) \
            + (S * self.q *np.exp(-self.q*tau) * stats.norm.cdf(-d1))/(r * (1-h)) \
            - (S * np.exp(-self.q * tau) * stats.norm.pdf(d1))/(2*r*tau*(1-h))\
            * (2*np.log(S/self.K)/(self.sigma * np.sqrt(tau)) - d1)

        dFdS = (1 - qQD) * (1 - np.exp(-q * tau) * stats.norm.cdf(-d1)) \
                + (np.exp(-q * tau) * stats.norm.pdf(d1))/(self.sigma * np.sqrt(tau))
        return -dFdh/dFdS
 


# class cheby_interp():
#     def __init__(self,y,transformation_func,x_min,x_max):
#         self.n = len(y) - 1 
#         self.a = [0] * len(y)
#         self.x_converting_cheby_func = transformation_func 
#         self.x_min = x_min
#         self.x_max = x_max 
#         for i in range(len(y)):
#             self.a[i] = self.calculate_params(i,y)
        


#     def get_values(self,new_values):
#         values = []
#         for z in new_values:
#             values.append(self.calculate_cheby_value_single(z))
        
#     def calculate_cheby_value_single(self,z):
#         '''
#         follow the update rules in equation (53). 
#         '''
#         b0 = self.a[self.n]
#         b1 = 0 
#         for k in range(self.n-1,-1,-1):
#             b1,b2 = b0,b1
#             b0 = self.a[k] + 2*z*b1 - b2 
#         qc = (b0-b2)*0.5
#         return qc 
#     def calculate_params(self, i,y_values):
#         params = 0 
#         for j in range(0,self.n+1):
#             temp = y_values[j] * np.cos(j*i*np.pi/self.n) 
#             if j ==0 or j == self.n:
#                 temp = 0.5*temp
#             params += temp
#         params += 2/self.n 
#         return params 


class cheby_interp:
    def __init__(self, ynodes, x_to_cheby, x_min, x_max):
        numpoints = len(ynodes)
        self.n = numpoints-1
        self.a = [0] * numpoints
        self.x_to_cheby = x_to_cheby
        self.x_min = x_min
        self.x_max = x_max
        for k in range(numpoints):
            self.a[k] = self.std_coeff(k, ynodes)

    @staticmethod
    def get_std_cheby_points(numpoints):
        i = np.arange(0, numpoints+1)
        return np.cos(i * np.pi/numpoints)

    def get_values(self, zv):
        ans = []
        if zv is float:
            to_cheby = [zv]
        else:
            to_cheby = zv
        if self.x_to_cheby is not None:
            to_cheby = self.x_to_cheby(zv, self.x_min, self.x_max)
        for z in to_cheby:
            ans.append(self.std_cheby_single_value(z))
        return ans

    def std_cheby_single_value(self, z):
        """z is the point to be valued between [-1, 1], n_y are the function values at Chebyshev points
        Iteration using Clenshaw algorithm"""
        b0 = self.a[self.n] * 0.5
        b1 = 0
        b2 = 0

        for k in range(self.n - 1, -1, -1):
            b1, b2 = b0, b1
            b0 = self.a[k] + 2 * z * b1 - b2
        return 0.5 * (b0 - b2)

    def std_coeff(self, k, n_y):
        ans = 0
        for i in range(0, self.n+1):
            term = n_y[i] * np.cos(i * k * np.pi / self.n)
            if i == 0 or i == self.n:
                term *= 0.5
            ans += term
        ans *= (2.0/self.n)
        return ans