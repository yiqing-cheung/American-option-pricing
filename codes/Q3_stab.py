# %load "testcode.py"
from spectral_collocation import *
from multiprocessing import Pool
import matplotlib.pyplot as plt
from itertools import product
import copy
import sys
import os
import pandas as pd
import numpy as np
import time
from matplotlib import pyplot as plt
from scipy.stats import norm

# Black-Scholes Functions for European Put Option Greeks
# Updated Black-Scholes Functions for European Put Option with Dividends
def d1_dividend(S, K, T, r, sigma, q):
    return (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

def d2_dividend(S, K, T, r, sigma, q):
    return d1_dividend(S, K, T, r, sigma, q) - sigma * np.sqrt(T)

def black_scholes_put_dividend(S, K, T, r, sigma, q):
    d1_val = d1_dividend(S, K, T, r, sigma, q)
    d2_val = d2_dividend(S, K, T, r, sigma, q)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2_val) - S * np.exp(-q * T) * norm.cdf(-d1_val)
    return put_price

def put_delta_dividend(S, K, T, r, sigma, q):
    return -np.exp(-q * T) * norm.cdf(-d1_dividend(S, K, T, r, sigma, q))

def put_gamma_dividend(S, K, T, r, sigma, q):
    return np.exp(-q * T) * norm.pdf(d1_dividend(S, K, T, r, sigma, q)) / (S * sigma * np.sqrt(T))

def put_vega_dividend(S, K, T, r, sigma, q):
    return S * np.exp(-q * T) * norm.pdf(d1_dividend(S, K, T, r, sigma, q)) * np.sqrt(T) 

def put_theta_dividend(S, K, T, r, sigma, q):
    term1 = -S * sigma * np.exp(-q * T) * norm.pdf(d1_dividend(S, K, T, r, sigma, q)) / (2 * np.sqrt(T))
    term2 = q * S * np.exp(-q * T) * norm.cdf(-d1_dividend(S, K, T, r, sigma, q))
    term3 = r * K * np.exp(-r * T) * norm.cdf(-d2_dividend(S, K, T, r, sigma, q))
    return (term1 + term2 - term3) 

def put_rho_dividend(S, K, T, r, sigma, q):
    return -T * K * np.exp(-r * T) * norm.cdf(-d2_dividend(S, K, T, r, sigma, q))  



# fuction for hiding print 
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class analyze_sc:
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


        # option_price under benchmark para:
        
        # Since the result doesn't change, we use pre-caculated prices here for convenience !
        # self.bm_pricer = spectral_collocation(*self.bm_para)
        # self.bm_price = self.bm_pricer.solve()
        self.bm_price = [9.569445298940671, 9.462492596167081]

        
    
    def accuary_test(self):
        '''
            Returns a list with parameter settings and precision indicators
        '''
        # Bulk Test
        para = copy.deepcopy(self.bm)
        K = 100
        r_range = [0.02, 0.06, 0.1]
        q_range = [0.04, 0.08, 0.12]
        S_range = [25,80,100,150,200]
        T_range = [1/12, 0.25, 0.5, 1]
        sigma_range = [0.1,0.3,0.5]
        para_list = [(5,2,4,8), (8,4,6,15), (21,6,10,41), (25,8,12,51), (31,16,16,61)]
        result_list = []

        for l, m, n, p in para_list:
            
            error_list = []
            print("testing parameter set ({}, {}, {}, {})".format(l, m, n, p))

            start = time.time()
            combinations = product(r_range, q_range, S_range, T_range, sigma_range)

            # 遍历所有组合
            for combination in combinations:
                r, q, S, T, sigma = combination
                temp_pricer = spectral_collocation(self.bm['type'], r,S,q,K,sigma,T,self.bm['method'],n,l,m,p)
                with HiddenPrints():
                    temp_price = temp_pricer.solve()
                    bm_price = spectral_collocation(self.bm['type'], r,S,q,K,sigma,T,self.bm['method'],self.bm['n'],self.bm['l'],self.bm['m'],self.bm['p']).solve()
                error_list.append(temp_price[0] - bm_price[0])
            end = time.time()

            rmse =  np.sqrt(np.mean([i*i for i in error_list]))
            rrmse = np.sqrt(np.mean([i*i for i in error_list]) / np.mean([(i + self.bm_price[0])*(i + self.bm_price[0]) for i in error_list])) 
            # In the paper, "MAE" means maximum absolute error and "MRE" means maximum relative error
            mae =  np.max([abs(i) for i in error_list])
            mre =  np.max([abs(i/bm_price[0]) for i in error_list])
            print("RMSE: ", rmse, ", RRMSE: ", rrmse, ", MAE: ", mae, ", MRE: ", mre, ", Cost Time: ", end-start)
            result_list += [(l, m, n, p), rmse, rrmse, mae, mre, 1/(end-start)]
        return result_list
    
    def demo_accuary_test_S(self, S = 100):
        para_list = [(5,2,4), (8,4,6), (21,6,10), (25,8,12), (31,16,16)]
        bm_para = copy.deepcopy(self.bm_para)
        print("Calculating the benchmark price...")
        with HiddenPrints():
            bm_para[2] = S 
            bm_price= spectral_collocation(*bm_para).solve()[0]
        print("bench mark price for S = {} is {}".format(S, bm_price))
        error_list = []
        for l, m, n in para_list:
            bm_para[9] = l
            bm_para[10] = m
            bm_para[8] = n
            with HiddenPrints():
                temp_pricer = spectral_collocation(*bm_para)
                temp_price = temp_pricer.solve()[0]
            error_list.append(temp_price-bm_price)
            print("With (l,m,n) = ({},{},{}), the option price is {}, error is {}".format(l, m, n, temp_price, temp_price-bm_price))
        x_axis = ["({},{},{})".format(i[0], i[1], i[2]) for i in para_list]
        plt.plot(x_axis, error_list)
        plt.show()
        df = pd.DataFrame({'x':x_axis, 'error':error_list})
        return df


    def accuary_test_S(self):
        print("We focus on the performance of hyper paremeters under three senario: ITM, ATM, OTM")
        print("ITM:")
        itm = self.demo_accuary_test_S(S = 50)
        print("ATM:")
        atm = self.demo_accuary_test_S(S = 100)
        print("OTM:")
        otm = self.demo_accuary_test_S(S = 150)
        with pd.ExcelWriter('accuary_test_S.xlsx') as writer:
            itm.to_excel(writer, sheet_name='ITM')
            atm.to_excel(writer, sheet_name='ATM')
            otm.to_excel(writer, sheet_name='OTM')
        return 0


    def demo_accuary_test_tau(self, T = 1/12):
        para_list = [(5,2,4), (8,4,6), (21,6,10), (25,8,12), (31,16,16)]
        bm_para = copy.deepcopy(self.bm_para)
        print("Calculating the benchmark price...")
        with HiddenPrints():
            bm_para[6] = T 
            bm_price= spectral_collocation(*bm_para).solve()[0]
        print("bench mark price for tau = {} is {}".format(T, bm_price))
        error_list = []
        for l, m, n in para_list:
            bm_para[9] = l
            bm_para[10] = m
            bm_para[8] = n
            with HiddenPrints():
                temp_pricer = spectral_collocation(*bm_para)
                temp_price = temp_pricer.solve()[0]
            error_list.append(temp_price-bm_price)
            print("With (l,m,n) = ({},{},{}), the option price is {}, error is {}".format(l, m, n, temp_price, temp_price-bm_price))
        x_axis = ["({},{},{})".format(i[0], i[1], i[2]) for i in para_list]
        plt.plot(x_axis, error_list)
        plt.show()
        df = pd.DataFrame({'x':x_axis, 'error':error_list})
        return df


    def accuary_test_tau(self):
        print("We focus on the performance of hyper paremeters under short and long term-to-maturity")
        print("Short Term:")
        df1 = self.demo_accuary_test_tau(T = 1/12)

        print("Long Term:")
        df2 = self.demo_accuary_test_tau(T = 1)

        with pd.ExcelWriter('accuary_test_tau.xlsx') as writer:
            df1.to_excel(writer, sheet_name='Short')
            df2.to_excel(writer, sheet_name='Long')

        return 0



    def demo_stability_test(self, para = "S", bp = 10):
        print("Test numerical stability for {}:\n".format(para))
        p = [0.05, 100, 0.05, 100, 0.25, 1]
        if para == "r": idx = 0
        elif para == "S": idx = 1
        elif para == "q": idx = 2
        elif para == "K": idx = 3
        elif para == "sigma": idx = 4
        elif para == "T": idx = 5
        para_start = copy.deepcopy(p[idx])
        prices = []
        x = []
        for i in range(-bp, bp+1):
            p[idx] = (1+i/10000) * para_start
            print("current value for {}: {}, started at {}".format(para, p[idx], para_start))
            with HiddenPrints():
                # To be modified!
                # temp_pricer = spectral_collocation(self.bm['type'], p[0],p[1], p[2], p[3], p[4], p[5],
                #                                    self.bm['method'],self.bm['n'],self.bm['l'],self.bm['m'],self.bm['p'])
                temp_pricer = spectral_collocation(self.bm['type'], p[0],p[1], p[2], p[3], p[4], p[5],
                                                   self.bm['method'],4,1,5,131)

                temp_price = temp_pricer.solve()
            prices.append(temp_price[0])
            x.append(copy.deepcopy(p[idx]))
        # plt.plot(x, prices)
        # plt.show()
        df = pd.DataFrame({para:x, 'opt_prices':prices})
        return df

    def stability_test(self, arg):
        # r_list, prices_r = self.demo_stability_test("r", bp = 10)

        # S_list, prices_S = self.demo_stability_test("S", bp = 10)

        # q_list, prices_q = self.demo_stability_test("q", bp = 10)

        # K_list, prices_K = self.demo_stability_test("K", bp = 10)

        # sigma_list, prices_sigma = self.demo_stability_test("sigma", bp = 10)

        # T_list, prices_T = self.demo_stability_test("T", bp = 10)
        
        return self.demo_stability_test(*arg)


    # def acurracy_test_tau():
    def demo_conver_test(self, para = "l", init_para = [5,1,4], start = 1, end = 100, precision = 5):
        ''' 
            para: the hyper-para to be changed; ex: "l"
            init_para: list; the initial value for (l,m,n)
            start: int; the first value of "para"
            start: int; first value of "para"
        '''
        print("Test convergence speed for {}:\n".format(para))
        if para == "l": idx = 0
        elif para == "m": idx = 1
        elif para == "n": idx = 2
        prices = []
        last_price = -999999
        conv = 100
        for i in range(start, end+1):
            if i%10 == 0:
                print("going through the {}th value".format(i))
            init_para[idx] = i
            with HiddenPrints():
                temp_pricer = spectral_collocation(self.bm['type'], self.bm["r"],self.bm["S"],self.bm["q"],self.bm["K"],
                                                self.bm["sigma"],self.bm["T"],self.bm['method'],init_para[2],init_para[0],init_para[1],131)
                temp_price = temp_pricer.solve()
            prices.append(temp_price[0])
            if abs(temp_price[0] - last_price) < 10**(-precision):
                conv = i
                print("Under the criteria 1e-{}, model converges when {} = {}".format(precision, para, conv))
                break
            last_price = temp_price[0]
        plt.plot(range(start, conv + 1), prices)        
        # plt.plot(range(start, end + 1), prices)
        plt.show()
        df = pd.DataFrame({para:list(range(start, conv + 1)), 'opt_price':prices})
        return df

    def convergence_test(self):
        print("Starting from (l,m,n) = (5,1,4), with p = 131\n")
        # To be modified!
        df_l = self.demo_conver_test("l", init_para=[5,1,4], start=1, end=100, precision=10)
        
        df_m = self.demo_conver_test("m", init_para=[5,1,4], start=1, end=100, precision=10)

        df_n = self.demo_conver_test("n", init_para=[5,1,4], start=1, end=100, precision=10)
        with pd.ExcelWriter('convergence_test.xlsx') as writer:
            df_l.to_excel(writer, sheet_name='L')
            df_m.to_excel(writer, sheet_name='M')
            df_n.to_excel(writer, sheet_name='N')
        return 0
    

    def sc_quick_solver(self, option_type = "american_put", r= 0.05, S =100, q=0.05, K=100,sigma=0.25,T = 1,method = "default",n=10,l =21,m=6,p=41):
        sc = spectral_collocation(option_type,r,S,q,K,sigma,T,method,n,l,m,p)
        with HiddenPrints():
            prices = sc.solve()
        return prices   


    def greeks_calculation(self, bp = 0.0001, S = 100, n = 10,l = 21, m = 6, p = 41):
        option_type,r,bm_S,q,K,sigma,T,method = self.bm_para[:8]
        option_value = self.sc_quick_solver(S = S)[0]
        delta = (self.sc_quick_solver(S = S+S*bp)[0] - self.sc_quick_solver(S = S-S*bp)[0])/(2*S*bp)
        gamma = (self.sc_quick_solver(S = S+S*bp)[0] + self.sc_quick_solver(S = S-S*bp)[0]-2*option_value)/((S*bp)**2)
        cn_pricer_sigma_p = self.sc_quick_solver(S = S, sigma = sigma+sigma*bp)[0]
        cn_pricer_sigma_m = self.sc_quick_solver(S = S,sigma = sigma-sigma*bp)[0] 
        cn_pricer_T_p = self.sc_quick_solver(S = S,T = T+T*bp)[0] 
        cn_pricer_T_m = self.sc_quick_solver(S = S,T = T-T*bp)[0]
        cn_pricer_r_p = self.sc_quick_solver(S = S,r = r+r*bp)[0]
        cn_pricer_r_m = self.sc_quick_solver(S = S,r = r-r*bp)[0]
        vega = (cn_pricer_sigma_p-cn_pricer_sigma_m)/(2*sigma*bp)
        theta = (cn_pricer_T_m-cn_pricer_T_p)/(2*T*bp)
        rho = (cn_pricer_r_p-cn_pricer_r_m)/(2*r*bp)
        greeks = {"option_value":option_value,"delta":delta,"gamma":gamma,"vega":vega,"theta":theta,"rho":rho}
        return greeks


    def greeks_plot(self, greeks=['option_value','delta','gamma','vega','theta','rho'],n = 10,l = 21, m = 6, p = 41):
        S_list = list(range(10,163,3))
        result = []
        print('Start Calculation')
        count = 1
        S_np = np.array(range(10,163,3))
        r = 0.05
        q = 0.05
        K = 100
        sigma = 0.25
        T = 1 
        european = black_scholes_put_dividend(S_np, K, T, r, sigma, q)
        for S in S_list:
            if count%10==0:
                print("Implementing the {}th calculation".format(count))
            result.append(self.greeks_calculation(S=S,n = n,l = l, m = m, p = p))
            count += 1

        greeks_df = pd.DataFrame()
        for greek in greeks:
            if greek == "option_value":
                european = black_scholes_put_dividend(S_np, K, T, r, sigma, q)
            elif greek == "delta":
                european = put_delta_dividend(S_np, K, T, r, sigma, q)
            elif greek == "gamma":
                european = put_gamma_dividend(S_np, K, T, r, sigma, q)
            elif greek == "vega":
                european =  put_vega_dividend(S_np, K, T, r, sigma, q)
            elif greek == "theta":
                european =   put_theta_dividend(S_np, K, T, r, sigma, q)
            elif greek == "rho":
                european =   put_rho_dividend(S_np, K, T, r, sigma, q)
            greeks_df["European"] = european
            greeks_df["American"] = [i[greek] for i in result] 
            greeks_df.to_csv("/Users/wangzhiyuan/Derivative Pricing/american put using spectral collocation/results/greeks/sc/{}.csv".format(greek))
            fig = plt.figure(figsize = (12,8),dpi=400) 
            plt.plot(S_np,european,c = "#BE8905",label = "European")
            plt.plot(S_list, [i[greek] for i in result], c = "#682D7D",marker = "o",label = "American")
            plt.title(f'Greeks: {greek}')
            plt.xlabel('S')
            plt.ylabel(f'{greek}')
            ax = plt.gca()
            ax.ticklabel_format(style='plain')
            plt.grid(color='lightgray')
            plt.legend() 
            plt.savefig("/Users/wangzhiyuan/Derivative Pricing/american put using spectral collocation/results/sc_greeks/{}.pdf".format(greek))
        return 0

    # def test_conver


if __name__ == '__main__':
    t = analyze_sc()
    t.greeks_plot()
    # t.accuary_test()
    # t.accuary_test_S()
    # t.accuary_test_tau()
    # t.convergence_test()

    # =================  stability_test using multiprocessing    ===================
    # inputs = [('r', 10), ('S', 10), ('q', 10), ('K', 10), ('sigma', 10), ('T', 10)]

    # with multiprocessing.Pool() as pool:
    #     results = pool.map(t.stability_test, inputs)

    # with pd.ExcelWriter('stability_test.xlsx') as writer:
    #     for i in range(len(inputs)):
    #         results[i].to_excel(writer, sheet_name=inputs[i][0])
