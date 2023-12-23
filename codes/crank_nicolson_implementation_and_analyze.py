#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 20:11:59 2023

@author: wangzhiyuan
"""
"""
@author: Wang Yuqian
"""


from spectral_collocation import *
 
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from multiprocessing import Pool
import matplotlib.pyplot as plt
import copy
import sys
import os
import numpy as np
import time
from matplotlib import pyplot as plt
import pandas as pd
# fuction for hiding print
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

class CNOptionSolver:
    def __init__(self, riskfree, dividend, volatility, strike, maturity, N=200, M=200, verbose=False):
        self.r = riskfree
        self.q = dividend
        self.sigma = volatility
        self.K = strike
        self.T = maturity
        self.Smax = 3 * strike
        self.S = []
        self.X = []
        self.A = []
        self.b = []
        self.N = N
        self.M = M
        self.price_history = [] #save the price history of every time step
        self.max_dt = self.T/self.M      #self.max_dt =  self.T/self.N
        self.USE_PSOR = False
        self.tol = 1e-5
        self.max_iter = 200
        self.omega = 1.2
        self.cached_dt = 0
        self.err = 0
        self.iter = 0
        self.verbose = verbose
        self.solvePDE_time = 0
        self.ave_solvePDE_time = 0
    def solve(self, S0):
        self.setInitialCondition()

        self.solvePDE()

        x = self.S.flatten()
        y = self.X.flatten()
        option_price = np.interp(S0, x, y)
        return option_price

    def solvePDE(self):
        t = self.T
        count = 0
        while t > 0:
            start_time = time.time()
            dt = min(t, self.max_dt)
            self.setCoeff(dt)
            if self.USE_PSOR:
                self.solvePSOR()
            else:
                self.solveLinearSystem()
            t -= dt

            end_time = time.time()  # End timing after computation ends
            self.solvePDE_time += end_time - start_time  # Accumulate the computation time
            count += 1
            self.price_history.append(self.X.copy())
            if self.verbose:
                print("t = ", t, " err = ", self.err, "iters = ", self.iter)
        self.ave_solvePDE_time = self.solvePDE_time/count

    def setInitialCondition(self):
        self.S = np.linspace(0, self.Smax, self.N)  # 相当于论文中的M，将Smax拆分
        self.A = np.zeros((self.N, self.N))
        self.b = np.zeros((self.N, 1))
        self.X = np.maximum(self.K - self.S, 0)


    def setCoeff(self, dt):
        N = self.N
        r = self.r
        q = self.q
        S = self.S
        X = self.X
        sigma = self.sigma
        dS = S[1] - S[0]
        for i in range(0, N-1):
            alpha = 0.25 * dt * (np.square(sigma*S[i]/dS) - (r - q) * S[i]/dS)
            beta = 0.5 * dt * (r + np.square(sigma * S[i]/dS))
            gamma = 0.25 * dt * (np.square(sigma*S[i]/dS) + (r - q) * S[i]/dS)
            if i == 0:
                self.b[i] = X[i] * (1 - beta)
                self.A[i][i] = 1 + beta
            else:
                self.b[i] = alpha * X[i-1] + (1 - beta) * X[i] + gamma * X[i+1]
                self.A[i][i-1] = -alpha
                self.A[i][i] = 1 + beta
                self.A[i][i+1] = -gamma
        self.A[-1][N-4] = -1
        self.A[-1][N-3] = 4
        self.A[-1][N-2] = -5
        self.A[-1][N-1] = 2
        self.b[-1] = 0

    def solveLinearSystem(self):
        self.X = np.linalg.solve(self.A, self.b)

    def solvePSOR(self):
        N = self.N
        iter = 0
        omega = self.omega
        self.err = 1000
        while self.err > self.tol and iter < self.max_iter:
            iter += 1
            x_old = self.X.copy()
            for i in range(N-1):
                self.X[i] = (1 - omega) * self.X[i] + omega * self.b[i] / self.A[i][i]
                self.X[i] -= self.A[i][i+1] * self.X[i+1] * omega / self.A[i][i]
                self.X[i] -= self.A[i][i-1] * self.X[i-1] * omega / self.A[i][i]

            #for last row, use boundary condition
            self.X[N-1] = (1 - omega) * self.X[i] + omega * self.b[i] / self.A[i][i]
            for j in range(N-4, N):
                self.X[N-1] -= self.A[N-1][j] * self.X[j] * omega / self.A[N-1][N-1]

            self.applyConstraint()
            self.err = np.linalg.norm(x_old - self.X)
            self.iter = iter

    def applyConstraint(self):
        self.X = np.maximum(self.X, self.K - self.S)

    def plotOptionPrices(self):
        """Plot the option prices against the stock prices."""
        x = np.array(self.S).flatten()  # Ensure x is a flat array
        y = np.array(self.X).flatten()  # Ensure y is a flat array
        plt.plot(x, y)
        plt.xlabel('Stock Price')
        plt.ylabel('Option Price')
        plt.title('Option Price-Stock Prices')
        plt.show()

    def plotPriceEvolution3D(self):
        """Plot the evolution of option value over time in 3D."""
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111, projection='3d')

        # Create a mesh grid for stock prices and time to maturity
        S_grid, T_grid = np.meshgrid(self.S, np.linspace(self.T, 0, len(self.price_history)))
        Z = np.array(self.price_history)
        # Plot the surface
        surf = ax.plot_surface(S_grid, T_grid, Z, cmap=plt.cm.viridis)
        fig.colorbar(surf, shrink=0.5, aspect=5)

        ax.set_xlabel('Underlying Price')
        ax.set_ylabel('Time to Maturity')
        ax.set_zlabel('Option Value')
        ax.set_title('Evolution of Option Value Over Time')

        plt.show()

    def get_solvePDE_time(self):
        """Return the time taken to execute solvePDE."""
        if self.solvePDE_time != 0:
            return [self.solvePDE_time, self.ave_solvePDE_time]
        else:
            print("solvePDE has not been run yet.")
            return None

    def saveResults(self, filename):
        """Save option price to csv"""
        results = pd.DataFrame(self.price_history, columns=[f'S{float(s)}' for s in self.S])
        results.to_csv(filename, index=False)
        print(f'Complete results saved to {filename}')

class CN_analysis:
    def __init__(self):
        '''
        # benchmark parameters
        self.bm_type = "american_put"
        self.bm_r = 0.05
        self.bm_S = 100
        self.bm_q = 0.05
        self.bm_K = 100
        self.bm_sigma = 0.25
        self.bm_T = 1
        self.bm_method = "default"
        self.bm_n = 64
        self.bm_l = 131
        self.bm_m = 16
        self.bm_p = 131
        '''
        self.bm = {'type' : "american_put",
                        'r':0.05,
                        'S':100,
                        'q':0.05,
                        'K':100,
                        'sigma':0.25,
                        'T':1,
                        'method':"default",
                        'n':64,
                        'l':131,
                        'm':16,
                        'p':131}
        self.bm_para = list(self.bm.values())
        self.bm_para = list(self.bm.values())
        self.bm_price = [9.569445298940671, 9.462492596167081]



    def accuary_test(self):
        '''
            Returns a list with parameter settings and precision indicators
        '''
        # Bulk Test
        para = copy.deepcopy(self.bm)
        K = 100
        r__q_range = [0.02, 0.04, 0.06, 0.08, 0.1]
        S_range = [25,50,80,90,100,110,120,150,175,200]
        T_range = [1/12, 0.25, 0.5, 0.75, 1]
        sigma_range = [0.1,0.2,0.3,0.4,0.5,0.6]
        para_list = [(100,100),(250,250),(500,500)]
        result_list = []
        #count = 0
        error = pd.DataFrame()
        accuracy_pd = pd.DataFrame()
        for n,m in para_list:

            error_list = []
            print("testing parameter set ({}, {})".format(n,m))

            start = time.time()
            for r,q,T,sigma in zip(r__q_range, r__q_range, T_range, sigma_range):
                temp_pricer = CNOptionSolver(r,q,sigma,K,T,N=n,M=m)
                temp_pricer.USE_PSOR = True
                temp_pricer.verbose = False
                for S in S_range:
                  with HiddenPrints():
                      temp_price = temp_pricer.solve(S)
                      bm_price = spectral_collocation(self.bm['type'], r,S,q,K,sigma,T,self.bm['method'],4,5,1,131).solve()
                  error_list.append(temp_price - bm_price[0])
            end = time.time()

            rmse =  np.sqrt(np.mean([i*i for i in error_list]))
            rrmse = np.sqrt(np.mean([i*i for i in error_list]) / np.mean([(i + self.bm_price[0])*(i + self.bm_price[0]) for i in error_list]))
            # In the paper, "MAE" means maximum absolute error and "MRE" means maximum relative error
            mae =  np.max([abs(i) for i in error_list])
            mre =  np.max([abs(i/bm_price[0]) for i in error_list])
            print("RMSE: ", rmse, ", RRMSE: ", rrmse, ", MAE: ", mae, ", MRE: ", mre, ", Cost Time: ", end-start)
            result_list += [(n,m), rmse, rrmse, mae, mre, 1/(end-start)]
            print(error_list)
            error[f'({n},{m})'] = error_list
            accuracy_pd[f'({n},{m})'] = [rmse,rrmse,mae,mre]
        accuracy_pd.index = ['RMSE','RRMSE','MAE','MRE']
        error.to_csv('Error.csv',index=True)
        accuracy_pd.to_csv('Accuracy.csv', index=True)
        return result_list

    def accuary_test2(self, S_list = [75,100,150]):
        para_list = [(50,50),(100,100),(250,250),(500,500)]
        bm_para = copy.deepcopy(self.bm_para)
        print("Calculating the benchmark price...")
        error = pd.DataFrame()
        for S in S_list:
          with HiddenPrints():
              bm_para[2] = S
          bm_price= spectral_collocation(*bm_para).solve()[0]
          print("bench mark price for S = {} is {}".format(S, bm_price))
          error_list = []
          for n,m in para_list:
              with HiddenPrints():
                  temp_pricer = CNOptionSolver(bm_para[1],bm_para[3],bm_para[5],bm_para[4],bm_para[6],N=n,M=m)
                  temp_pricer.USE_PSOR = True
                  temp_pricer.verbose = False
                  temp_price = temp_pricer.solve(S)
              error_list.append(abs(temp_price-bm_price))
              print("With (n,m) = ({},{}), the option price is {}, error is {}".format(n, m, temp_price, temp_price-bm_price))
          fid = plt.figure(figsize = (10,6),dpi=400)
          x_axis = ["({},{})".format(i[0], i[1]) for i in para_list]
          if S < bm_para[4]:
            sns.lineplot(x=x_axis, y=error_list, label='ITM',color = "red")
            print('Minimum error under ITM:',min(error_list))
            error['ITM'] = error_list
          elif S == bm_para[4]:
            sns.lineplot(x=x_axis, y=error_list, label='ATM',color = "blue")
            print('Minimum error under ATM:',min(error_list))
            error['ATM'] = error_list
          else:
            sns.lineplot(x=x_axis, y=error_list, label='OTM',color = "green")
            print('Minimum error under OTM:',min(error_list))
            error['OTM'] = error_list
        plt.title('Performance of hyper paremeters under three senario: ITM, ATM, OTM')
        plt.xlabel('Parameters')
        plt.ylabel('Error')
        plt.grid(color='lightgray')
        plt.savefig("/Users/wangzhiyuan/Derivative Pricing/american put using spectral collocation/results/CN Plots/Accuracy_TMS.pdf")
        error.index = para_list
        return error

    def stability_under_parameters(self, para = "S", delta = 10):
        print("Test numerical stability for {}:\n".format(para))
        parameters = [0.05, 100, 0.05, 100, 0.25, 1]
        if para == "r": idx = 0
        elif para == "S": idx = 1
        elif para == "q": idx = 2
        elif para == "K": idx = 3
        elif para == "sigma": idx = 4
        elif para == "T": idx = 5
        para_start = copy.deepcopy(parameters[idx])
        prices_sc = []
        prices_cn = []
        x_axis = []
        bp = 1/10000
        price_comparison = pd.DataFrame()
        for i in range(-delta, delta+1):
            parameters[idx] = (1+i*bp) * para_start
            # with HiddenPrints():
            #     sc_pricer = spectral_collocation(self.bm['type'], parameters[0],parameters[1], parameters[2], parameters[3], parameters[4], parameters[5],
            #                                        self.bm['method'],4,5,1,131)
            #     sc_price = sc_pricer.solve()
            # prices_sc.append(sc_price[0])
            cn_pricer = CNOptionSolver(parameters[0],parameters[2],parameters[4],parameters[3],parameters[5],N=250,M=250)
            cn_pricer.USE_PSOR = True
            cn_pricer.verbose = False
            cn_price = cn_pricer.solve(parameters[1])
            prices_cn.append(cn_price)
            x_axis.append(copy.deepcopy(parameters[idx]))
        #fig = plt.figure(figsize = (8,6))
        #sns.lineplot(x=x_axis, y=prices_cn, label='Crank-Nicolson')
        #sns.lineplot(x=x_axis, y=prices_sc, label='Spectral Collocation')
        # plt.title(f'Pricers stability under parameter {para}')
        # plt.xlabel(f'{para}')
        # plt.ylabel('Price')
        # plt.grid(color='lightgray')
        # plt.show()
        price_comparison['Crank-Nicolson'] = prices_cn
        #price_comparison['Spectral Collocation'] = prices_sc
        price_comparison.index = [f'{para}={(1+i*bp) * para_start}' for i in range(-delta, delta+1)]
        print(price_comparison)
        price_comparison.to_csv('/Users/wangzhiyuan/Derivative Pricing/american put using spectral collocation/results/cn_stability/Stability_under_{}.csv'.format(para),index = True)
        bias_cn = (prices_cn[-1] - prices_cn[0])/prices_cn[0]
        bias_sc = 0
        return price_comparison, [bias_cn,bias_sc]

    def stability_test(self):
      bias_r = self.stability_under_parameters("r", delta = 10)[1]
      bias_S = self.stability_under_parameters("S", delta = 10)[1]
      bias_q = self.stability_under_parameters("q", delta = 10)[1]
      bias_sigma = self.stability_under_parameters("sigma", delta = 10)[1]
      bias_T = self.stability_under_parameters("T", delta = 10)[1]
      bias_K = self.stability_under_parameters("K", delta = 10)[1]
      bias = pd.DataFrame(
          {
          'r':bias_r,
          'S':bias_S,
          'q':bias_q,
          'sigma':bias_sigma,
          'T':bias_T,
          "K":bias_K
      },index=['Crank-Nicolson','Spectral Collocation'])
      return bias

    def convergence_under_nm(self, para, para_list = [250,250], start = 100, end = 1000, precision = 5):
        '''
            para: the hyper-para to be changed;
            init_para: list; the initial value for (N,M)
            start: int; the first value of "para"
            start: int; first value of "para"
        '''
        print("Test convergence speed for {}:\n".format(para))
        if para == "N": idx = 0
        elif para == "M": idx = 1
        prices = []
        last_price = -999999
        conv = 100
        for i in range(start, end+1, 25):
            if i%100 == 0:
                print("going through the {}th value".format(i))
            para_list[idx] = i
            with HiddenPrints():
              cn_pricer = CNOptionSolver(self.bm["r"],self.bm["q"],self.bm["sigma"],self.bm["K"],self.bm["T"],para_list[0],para_list[1])
              cn_pricer.USE_PSOR = True
              cn_pricer.verbose = False
              cn_price = cn_pricer.solve(self.bm['S'])
            prices.append(cn_price)
            if abs(cn_price - last_price) < 1*(10**(-precision)):
                conv = i
                print("Under the criteria 1e-{}, model converges when {} = {}".format(precision, para, conv))
                break
            last_price = cn_price
        plt.figure(figsize = (12,8),dpi=400)  
        sns.lineplot(x=range(start, conv+1, 25), y=prices, label='Crank-Nicolson')
        plt.title(f'Pricers convergence test under parameter {para}')
        plt.xlabel(f'{para}')
        plt.ylabel('Price')
        ax = plt.gca()
        ax.ticklabel_format(style='plain')
        plt.grid(color='lightgray')
        plt.legend() 
        plt.savefig("/Users/wangzhiyuan/Derivative Pricing/american put using spectral collocation/results/CN Plots/convergence{}.pdf".format(para))
        plt.show()
        return prices, conv

    def convergence_test(self):
      try:
        print("Starting from (N,M) = (250,250)")
        prices_n, conv_n = self.convergence_under_nm("N", precision=5)

        prices_m, conv_m = self.convergence_under_nm("M", precision=5)
        Convergence = pd.DataFrame(
            {
                'Convergence Speed': [f'N={conv_n}', f'M={conv_m}']
            }
        )
        Convergence.index = ['When M=500, model convergence at', 'When N=500, model convergence at']
        Convergence.to_csv('Convergence Speed.csv', index = True)
        return Convergence
      except: print('Not converge in the given range')

    def greeks_calculation(self, pct_change = 0.0001, S = 100, n = 250, m = 250):
      r = self.bm['r']
      q = self.bm['q']
      sigma = self.bm['sigma']
      K = self.bm['K']
      T = self.bm['T']
      cn_pricer = CNOptionSolver(r, q, sigma, K, T, n, m)
      cn_pricer.USE_PSOR = True
      cn_pricer.verbose = False
      option_value = cn_pricer.solve(S)
      print("Under S = {},option value = {}".format(S,option_value))
      delta = (cn_pricer.solve(S+S*pct_change)-cn_pricer.solve(S-S*pct_change))/(2*S*pct_change)
      gamma = (cn_pricer.solve(S+S*pct_change)+cn_pricer.solve(S-S*pct_change)-2*cn_pricer.solve(S))/((S*pct_change)**2)
      cn_pricer_sigma_p = CNOptionSolver(r, q, sigma+sigma*pct_change, K, T, n, m)
      cn_pricer_sigma_p.USE_PSOR = True
      cn_pricer_sigma_p.verbose = False
      cn_pricer_sigma_m = CNOptionSolver(r, q, sigma-sigma*pct_change, K, T, n, m)
      cn_pricer_sigma_m.USE_PSOR = True
      cn_pricer_sigma_m.verbose = False
      cn_pricer_T_p = CNOptionSolver(r, q, sigma, K, T+T*pct_change, n, m)
      cn_pricer_T_p.USE_PSOR = True
      cn_pricer_T_p.verbose = False
      cn_pricer_T_m = CNOptionSolver(r, q, sigma, K, T-T*pct_change, n, m)
      cn_pricer_T_m.USE_PSOR = True
      cn_pricer_T_m.verbose = False
      cn_pricer_r_p = CNOptionSolver(r+r*pct_change, q, sigma, K, T, n, m)
      cn_pricer_r_p.USE_PSOR = True
      cn_pricer_r_p.verbose = False
      cn_pricer_r_m = CNOptionSolver(r-r*pct_change, q, sigma, K, T, n, m)
      cn_pricer_r_m.USE_PSOR = True
      cn_pricer_r_m.verbose = False
      vega = (cn_pricer_sigma_p.solve(S)-cn_pricer_sigma_m.solve(S))/(2*sigma*pct_change)
      theta = (cn_pricer_T_m.solve(S)-cn_pricer_T_p.solve(S))/(2*T*pct_change)
      up = cn_pricer_r_p.solve(S)
      print("r_up",up)
      down = cn_pricer_r_m.solve(S) 
      print("r_down",down) 
      rho = (cn_pricer_r_p.solve(S)-cn_pricer_r_m.solve(S))/(2*r*pct_change)
      greeks = {"option_value":option_value,"delta":delta,"gamma":gamma,"vega":vega,"theta":theta,"rho":rho}
      return greeks
    def greeks_plot(self, greeks=['option_value','delta','gamma','vega','theta','rho'],n = 250, m = 250):
      S_list = range(10,161,3)
      result = []
      print('Start Calculation')
      for S in S_list:
        if S%10==0:
          print(f'Calculation for S={S}')
        result.append(self.greeks_calculation(S=S, n=n, m=m))
      for greek in greeks:
        fig = plt.figure(figsize = (12,8),dpi=400) 
        plt.plot(S_list, [i[greek] for i in result],c = "#682D7D",marker = "o")
        greek_df = pd.DataFrame()
        greek_df["S"] = S_list
        greek_df["{}".format(greek)] = [i[greek] for i in result]
        greek_df.to_csv('/Users/wangzhiyuan/Derivative Pricing/american put using spectral collocation/results/greeks/cn/{}.csv'.format(greek))
        plt.title(f'Greeks: {greek}')
        plt.xlabel('S')
        plt.ylabel(f'{greek}')
        ax = plt.gca()
        ax.ticklabel_format(style='plain')
        plt.grid(color='lightgray')
        plt.legend() 
        plt.savefig("/Users/wangzhiyuan/Derivative Pricing/american put using spectral collocation/results/CN Plots/{}.pdf".format(greek))
        
if __name__ == '__main__':
    test = CN_analysis()
    #test.accuary_test()
    #test.accuary_test2()
    test.stability_test()
    #test.convergence_test()
    #test.greeks_plot( n=250, m=250)
    #para_list = [(50,50),(100,100),(250,250),(500,500)] 
    # x_axis = ["({},{})".format(i[0], i[1]) for i in para_list]
    # plt.figure(figsize = (10,6),dpi=400)  
    # plt.plot(x_axis,itm,color = "green",marker = "o",label = "ITM") 
    # plt.plot(x_axis,atm,color = "blue",marker = "^",label = "ATM") 
    # plt.plot(x_axis,otm,color = "red",linestyle='dashed',label = "OTM") 
    # plt.title('Performance of hyper paremeters under three senario: ITM, ATM, OTM')
    # plt.xlabel('Parameters')
    # plt.ylabel('Error')
    # plt.legend()
    # plt.grid(color='lightgray')
    # plt.savefig("/Users/wangzhiyuan/Derivative Pricing/american put using spectral collocation/results/CN Plots/Accuracy_TMS.pdf")
