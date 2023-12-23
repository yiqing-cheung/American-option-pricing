
"""
Created on Sat Nov  5 10:50:35 2023

@author: wangzhiyuan
"""
import numpy as np 
import math
from scipy.stats import norm
import scipy 
import time 
import utils #this is self-defined scripts that support qd+,Gauss-Legendre integration, Chebyshev Poly. Interp, 
import scipy.stats as stats
import numpy.polynomial.legendre as legendre
import numpy.linalg  

class spectral_collocation():
    def __init__(self,option_type,r,S,q,K,sigma,T,method = "default",collocation_num_n=12,quadrature_num_l =24,iteration_num_m=200,itegral_num_p=48) :
        ##################Option Params##########################
        self.option_type = option_type 
        self.r = r #risk free rate 
        self.q = q #dividend 
        self.sigma = sigma  #volatility
        self.K = K  #Strike price 
        self.S = S #Spot Stock price 
        self.T = T  #maturity 
        ##################Algo Params############################
        self.quadrature_num = quadrature_num_l
        self.collocation_num = collocation_num_n #parameter n 
        #self.integral_num = 2*self.quadrature_num
        self.integral_num = itegral_num_p
        self.max_iters = iteration_num_m  #parameter m 

        self.iter_tol = 1e-11

        #self.shared_B_old = []
        self.shared_tau = []
        self.chebyshev_B = [] 
        self.integration_B = [] 

        self.tau_max = self.T
        self.tau_min = 0
        self.method = method

                # points and weights for Guassian integration
        self.iter_records = []
        self.error = 1000000
        self.num_iters = 0
    def solve(self): 

        if self.option_type == "american_put":
            if self.method == "default":
                self.method = "spectral collocation"
                price, cpu_time = self.pricing_american_put() 
                #print(price)
            elif self.method in ["crank nikoson",'binomial']:
                price, cpu_time = self.pricing_american_put() 
            else:
                raise ValueError("Method not specified for American Put")
             

        if self.option_type == "american_call":
            if self.q == 0:
                price, cpu_time = self.pricing_european_call(self) 
            else:
                price, cpu_time = self.pricing_american_call(self) 

        if self.option_type == "eurupean_call":
            price, cpu_time = self.pricing_european_call()

        if self.option_type == "european_put":
            price, cpu_time = self.pricing_european_put() 
        
        #self.display(price,cpu_time) 
        return price,self.pricing_european_put()[0],cpu_time
    ##############################Utils function############################
    def display(self, price,cpu_time):
        '''
        display the pramameters and price. 
        '''
        print("Option Pricing Summary".center(50, '-'))
        print(f"Stock Price (S0): {self.S}")
        print(f"Strike Price (K): {self.K}")
        print(f"Time to Expiration (T): {self.T}")
        print(f"Risk-free Rate (r): {self.r}")
        print(f"Volatility (sigma): {self.sigma}")
        print(f"Dividend Yield (q): {self.q}")
        print(f"Option Type: {self.option_type.capitalize()}")
        print(f"Option Price: {price}")
        print(f"CPU Time: {cpu_time} seconds")
        if self.method == "spectral collocation":
            print(f"Collocation Number (n): {self.collocation_num} ")
            print(f"Quafrature Number (l): {self.quadrature_num}")
            print(f"Integration Number (l): {self.integral_num}")
            print(f"Iteration Number (m): {self.max_iters}")
        print('-'*50)

    def time_cpu(func):
        '''
        define a decorator to calculate the cpu time of running a function. 
        '''
        def wrapper(*args, **kwargs):
            start_time = time.process_time()  # Get the starting CPU time
            result = func(*args, **kwargs)    # Call the original function
            end_time = time.process_time()    # Get the ending CPU time
            
            #print(f"{func.__name__} executed in {end_time - start_time:.6f} CPU seconds.")
            
            return result, end_time - start_time 

        return wrapper
    ##############################American Put############################
    #-----------------------Spectral Collocation--------------------------#
    #Step 1: set collocation points. 
    def set_collocation_points(self):
        '''
        set collocation points and the interpolation points for H(x)
        '''
        def set_std_cheby_points(numpoints):
            i = np.arange(0, numpoints+1)
            return np.cos(i * np.pi/numpoints)
        cheby_points = set_std_cheby_points(self.collocation_num) 

        #h_x_points = np.sqrt(self.tau_max) * (1+cheby_points) /2 #equation (50) : get the interpolation points for H(x_i). 
        #self.shared_tau = np.square(h_x_points) #equation (54): get the collocation points for fix point iteration. 
        self.shared_tau = self.tau_max * np.square(1+cheby_points) / 4
 

    #step 2: use fixed point iteration to get exercise boundary values at collocation points 
    def get_exercise_boundary(self):
        self.initial_value()
        ##################################
        iter_count = 0
        iter_err = 1
        #print("tau in get exercise bound", self.shared_tau) #no prob 
 
        while (iter_err > self.iter_tol) and (iter_count < self.max_iters):

            iter_count += 1
            B_old = self.chebyshev_B.copy()

            self.chebyshev_B = self.iterate_once(self.shared_tau, B_old)
            #print("tau after once iter",self.shared_tau)
            self.chebyshev_B_old = B_old
            iter_err = self.norm1_error(B_old, self.chebyshev_B)
            #print("  iter = {0}, err = {1}".format(iter_count, self.norm1_error(B_old, self.chebyshev_B)))
            self.iter_records.append((iter_count, iter_err))

        self.error = iter_err
        self.num_iters = iter_count
    def norm1_error(self, x1, x2):
        x1 = np.array(x1)
        x2 = np.array(x2)
        return numpy.linalg.norm(np.abs(x1 - x2))

    
    #---------------------helper functions for step 2-----------------------# 
    #1. intial_value(): set initial value for each B(tau_i) 
    #2. define iterate rules for each iteration for each tau_i. 
    #3. func 2 to an interation function for each iteration and every tau_i 
    #4. dminus(tau,z) in page 4
    #5. dplus(tau,z) in page 4 
    #6. K1 in equation (18)
    #7. K2 in equation (19)
    #8. K3 in equation (20)
    #-----------------------------------------------------------------------#
    #func 9-16 are just some simple combinations of normal pdf and cdf and 
    # {a}d_{b}, a in [+,-],b in [+,-].  
    #These combinations are simple but convenient for later uses.
    #-----------------------------------------------------------------------# 
    #9. pdf_plus_dmius(tau,z) (\phi(d_{-}(\tau,z))) 
    #10. pdf_minus_dmius(tau,z) (\phi(-d_{-}(\tau,z))) 
    #11. pdf_plus_dplus(tau,z) (\phi(d_{+}(\tau,z))) 
    #12. pdf_minus_dplus(tau,z) (\phi(-d_{+}(\tau,z)))    
    #13. cdf_plus_dminus(tau,z)
    #14. cdf_minus_dminus(tau,z)
    #15. cdf_plus_dplus(tau,z)
    #16. cdf_minus_dplus(tau,z) 
    #-----------------------------------------------------------------------#
    #func 17-22 are the main components for the iteration rule. 
    #-----------------------------------------------------------------------# 
    #17. N(tau,B) from equation (21)
    #18. D(tau,B) from equation (22) 
    #19. Nprime in equation (39)
    #20. Dprime in equation (40)  
    #21. f(tau,B) in equation (37) 
    #22. f_prime(tau,B) in equation (38)
    #-----------------------------------------------------------------------#
    #other helper funcs
    #-----------------------------------------------------------------------# 
    #23. 

    def initial_value(self):
        #this need to be revised and checked.
        """get initial guess for all tau_i using QD+ algorithm"""
        qdplus_initializer = utils.QDplus(self.r, self.q, self.sigma, self.K )
        res = []
        for tau_i in self.shared_tau:
            res.append(qdplus_initializer.compute_exercise_boundary(tau_i))
        self.chebyshev_B = res
        #print("initial B",self.chebyshev_B)
        
        #self.shared_B0 = res.copy()


       
    def iterate_once_foreach_tau(self, tau_i, B_i):
        eta = 0.5
        f_and_fprime = self.f_and_fprime(tau_i, B_i)
        f = f_and_fprime[0]
        fprime = f_and_fprime[1]
        #print("tau in iteration ",tau_i)
        #print("tau_i = ", tau_i, "analy fprime = ", f_and_fprime[1])
        if tau_i == 0:
            B_i = self.B_zero()
        else:
            B_i += eta * (B_i - f) / (fprime - 1)
        return B_i
    
    def iterate_once(self, tau, B):
        """the for-loop can be parallelized"""
        B_new = []
        #print("tau before iteration",tau)
        for i in range(len(tau)):
            #print("tau before self int", tau[i])
            B_i = self.iterate_once_foreach_tau(tau[i], B[i])
            B_new.append(B_i)
        return B_new   
    
    
    def dminus(self, tau, z):
        return (np.log(z) + (self.r - self.q)*tau - 0.5 * self.sigma * self.sigma * tau)/(self.sigma * np.sqrt(tau))

    def dplus(self, tau, z):
        return (np.log(z) + (self.r - self.q)*tau + 0.5 * self.sigma * self.sigma * tau)/(self.sigma * np.sqrt(tau))    
    


    def integrand_in_K3(self, tau, B_tau, u, B_u):
        #ans = 0
        return np.exp(self.r * u)/(self.sigma * np.sqrt(tau - u)) * self.pdf_plus_dminus(tau-u, B_tau/B_u)
    def K3(self, tau, B):
        return self.numerical_integration(self.integrand_in_K3,B, tau,   self.quadrature_num)
    
    def K1_plus_K2(self, tau, B):
        return self.numerical_integration(self.integrand_in_K1_plus_K2, B, tau, self.quadrature_num)

    def integrand_in_K1_plus_K2(self, tau, B_tau, u, B_u):
        return np.exp(self.q * u) * self.cdf_plus_dplus(tau - u, B_tau/B_u) + np.exp(self.q * u)/(self.sigma * np.sqrt(tau - u)) * self.pdf_plus_dplus(tau-u, B_tau/B_u)


    
    def pdf_plus_dminus(self, tau, z):
        return 0 if tau == 0 else stats.norm.pdf(self.dminus(tau, z))

    def pdf_plus_dplus(self, tau, z):
        return 0 if tau == 0 else stats.norm.pdf(self.dplus(tau, z)) 

    def pdf_minus_dminus(self, tau, z):
        return 0 if tau == 0 else stats.norm.pdf(-self.dminus(tau, z))

    def pdf_minus_dplus(self, tau, z):
        return 0 if tau == 0 else stats.norm.pdf(-self.dplus(tau, z))
    
    def cdf_minus_dminus(self, tau, z):
        if tau == 0 and z > 1:
            #dminus is negative inf in this case 
            return 0
        elif tau == 0 and z <= 1:
            #dminus is  inf in this case 
            return 1
        else:
            return stats.norm.cdf(-self.dminus(tau, z))

    def cdf_plud_dminus(self, tau, z):
        if tau == 0 and z > 1:
            return 1
        elif tau == 0 and z <= 1:
            return 0
        else:
            return stats.norm.cdf(self.dminus(tau, z))

    def cdf_minus_dplus(self, tau, z):
        if tau == 0 and z > 1:
            return 0
        elif tau == 0 and z <= 1:
            return 1
        else:
            return stats.norm.cdf(-self.dplus(tau, z))

    def cdf_plus_dplus(self, tau, z):
        if tau == 0 and z > 1:
            return 1
        elif tau == 0 and z <= 1:
            return 0
        else:
            return stats.norm.cdf(self.dplus(tau, z))
        
    def N(self, tau, B):
        K3 = self.K3(tau, B)
        return self.pdf_plus_dminus(tau, B/self.K)/(self.sigma * np.sqrt(tau)) + self.r * K3


    def D(self, tau, B):
        K12 = self.K1_plus_K2(tau, B)
        return self.pdf_plus_dplus(tau, B/self.K)/(self.sigma * np.sqrt(tau)) + self.cdf_plus_dplus(tau, B/self.K) + self.q * (K12)
    
    def intergrand_in_Nprime(self, tau, B_tau, u, B_u):
        tau = max(1e-10, tau) #to avoid the case tau=0, d would go to inf. 
        return np.exp(self.r * u) * self.dminus(tau-u, B_tau/B_u) / (B_tau * self.sigma * self.sigma * (tau - u)) * self.pdf_plus_dminus(tau-u, B_tau/B_u)
    def Nprime(self, tau, Q):
        tau = max(1e-10, tau)
        return -self.dminus(tau, Q/self.K) * self.pdf_plus_dminus(tau, Q/self.K) / (Q * self.sigma * self.sigma * tau) - self.r * self.numerical_integration(self.intergrand_in_Nprime,  Q,tau, self.quadrature_num)
 

    def intergrand_in_Dprime(self, tau, B_tau, u, B_u):
        tau = max(1e-10, tau)
        return np.exp(self.q * u) * self.pdf_plus_dplus(tau-u, B_tau/B_u)/(self.sigma * np.sqrt(tau - u) * B_tau) * (1 - self.dplus(tau - u, B_tau/B_u)/(self.sigma * np.sqrt(tau - u)))
    def Dprime(self, tau, Q):
        tau = max(1e-10, tau)
        return self.pdf_plus_dplus(tau, Q/self.K)/(self.sigma * np.sqrt(tau) * Q) * (1 - self.dplus(tau, Q/self.K)/(self.sigma * np.sqrt(tau))) + self.q * self.numerical_integration(self.intergrand_in_Dprime,  Q,tau, self.quadrature_num)
 


    def f_and_fprime(self, tau_i, B_i):
        #define f and fprime in one func, because they both rely on N and D, seperate definition would result in repeated caculations. 
        if tau_i == 0:
            return [self.B_zero(), 1]
        N = self.N(tau_i, B_i)  
        D = self.D(tau_i, B_i) 
        f = self.K * np.exp(-tau_i * (self.r - self.q)) * N / D
        Ndot = self.Nprime(tau_i, B_i)
        Ddot = self.Dprime(tau_i, B_i)
        fprime = self.K * np.exp(-tau_i * (self.r - self.q)) * (Ndot / D - Ddot * N / (D * D))
        return [f, fprime]

    def compute_fprime_numerical(self, tau_i, B_i, B_i_old):
        if B_i == B_i_old:
            return 0
        up_res = self.f_and_fprime(tau_i, B_i)
        down_res = self.f_and_fprime(tau_i, B_i_old)
        f_up = up_res[0]
        f_down = down_res[0]
        return (f_up - f_down)/(B_i - B_i_old)
    
 
    
    def B_zero(self):
        return self.K*min(1,self.r/self.q)

    #step 3: use chebyshev interpolation to get the values of points of the integration points 
    def u_and_Bu_for_integration(self,tau,num_points):
        #use chebyshev interp to cover the points for integration.also generates the integral weights 
        gauss_legendre = legendre.leggauss(num_points)
        self.quadrature_weights = gauss_legendre[1]
        self.y = gauss_legendre[0]
        X = self.K * min(1,self.r/self.q) 
        H = np.square(np.log(np.array(self.chebyshev_B)/X))
        self.integration_u = tau  - tau * np.square(1+self.y)/4 
        # print(H) 
        chebyshev_interp = utils.cheby_interp(H,self.cheby_transformation,self.tau_min,self.tau_max) 
        self.integration_B = chebyshev_interp.get_values(self.integration_u)
        # print(self.integration_B) 
        self.integration_B = np.exp(-np.sqrt(np.maximum(0.0, self.integration_B))) * X

    #step 4: use gauss legendre quadrature rule to get the american put option value 
    def numerical_integration(self,integrand,S,tau,num_points):
        self.u_and_Bu_for_integration(tau,num_points) 
        integral = 0
        for i in range(len(self.integration_u)):
            temp = integrand(tau,S,self.integration_u[i],self.integration_B[i])*self.quadrature_weights[i] * (tau/2)*(1+self.y[i]) 
            integral += temp 
        return integral
    
    def intergrand_americanput(self,tau,S,u,Bu):
        return self.r * self.K * np.exp(-self.r * (tau - u)) * self.cdf_minus_dminus(tau - u, S / Bu) - self.q * S * np.exp(-self.q * (tau - u)) * self.cdf_minus_dplus(tau-u, S/Bu)
 

    def get_american_put_with_excercise_boundary(self, tau, s0, r, q, sigma, K):
        v,cpu_time = self.pricing_european_put() 
        v_int = self.numerical_integration(self.intergrand_americanput,  s0,tau, self.integral_num)
 
        return v + v_int
 
    def cheby_transformation(self, x, x_min, x_max):
        # x in [x_min, x_max] is transformed to [-1, 1]
        return np.sqrt(4 * (x - x_min) / (x_max - x_min)) - 1
    
 
    
    #combine the above four steps into a function. 
    @time_cpu 
    def pricing_american_put(self,method="spectral collocation"):
        '''
        pricing american put using spectral collocation method 
        '''
        if method == "spectral collocation":
            self.set_collocation_points()
            self.get_exercise_boundary() 
            return self.get_american_put_with_excercise_boundary(self.T,self.S,self.r,self.q,self.sigma,self.K)
             
        if method == "crank nikoson":
            pass 

        

        

    ##############################American Call############################
    #------------------------Put-Call Symmetry----------------------------#
    @time_cpu
    def pricing_american_call(self):
        '''
        Use american put call symmetry to get the price. 
        '''


        pass

    ##############################European Call############################
    #-----------------------Black-Scholes Formula-------------------------#
    @time_cpu
    def pricing_european_call(self):
        '''
        Use B-S formula to get european call 
        '''
        start_time = time.process_time()
        d1 = (math.log(self.S / self.K) + (self.r - self.q + 0.5 * self.sigma**2) * self.T) / (self.sigma * math.sqrt(self.T))
        d2 = d1 - self.sigma * math.sqrt(self.T)
        return self.S * math.exp(-self.q * self.T) * norm.cdf(d1) - self.K * math.exp(-self.r * self.T) * norm.cdf(d2)
  
    ##############################European Put############################
    #-----------------------Black-Scholes Formula-------------------------#
    @time_cpu
    def pricing_european_put(self): 
        '''
        Use put-call parity to get european put
        '''
        #d1 = np.log(self.S * np.exp((self.r-self.q)*self.T)/self.K)/(self.sigma * np.sqrt(self.T)) + 0.5*self.sigma * np.sqrt(self.T)
        d1 = (math.log(self.S / self.K) + (self.r - self.q + 0.5 * self.sigma**2) * self.T) / (self.sigma * math.sqrt(self.T))
        d2 = d1 - self.sigma * math.sqrt(self.T)
        return self.K * math.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * math.exp(-self.q * self.T) * norm.cdf(-d1)


         


