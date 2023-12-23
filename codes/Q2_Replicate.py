

from spectral_collocation import *
from multiprocessing import Process
def single_delta(S,delta_S=0.0001):
    K= 100 
    r = 0.05
    q = 0.05 
    sigma = 0.25 
    T = 1.0
    l = 131
    m=16
    n=64
    p=131 
    
    S_bumped = S + delta_S
    pricer = spectral_collocation(option_type = "american_put",r=r,S=S,q=q,K=K,sigma =sigma,T=T,method = "default",collocation_num_n=n,quadrature_num_l =l,iteration_num_m=m,itegral_num_p=p)
    price1 =  pricer.solve()[0]
    pricer = spectral_collocation(option_type = "american_put",r=r,S=S_bumped,q=q,K=K,sigma =sigma,T=T,method = "default",collocation_num_n=n,quadrature_num_l =l,iteration_num_m=m,itegral_num_p=p)
    price2 =  pricer.solve()[0]
    delta = (price2 - price1)/delta_S  
    f = open('/Users/wangzhiyuan/Derivative Pricing/american put using spectral collocation/results/greeks/delta/delta.txt', 'a')
    f.write("{}".format(delta))
    f.write("\n") 

    return delta

if __name__ == '__main__':
    '''
    # unit test one for valuing American option
    r = 0.04     # risk free
    q = 0.01      # dividend yield
    K = 100.0       # strike
    S = 80.0        # underlying spot
    sigma = 0.2     # volatility
    T = 3.0         # maturity
     

    solver = spectral_collocation("american_put",r,S, q,K, sigma,  T)
    solver.use_derivative = False
    solver.iter_tol = 1e-5
    solver.max_iters = 20
    price = solver.solve()   # t and S
   
    '''

    #replicate table 2 
    #S = 100
    K= 100 
    r = 0.05
    q = 0.05 
    sigma = 0.25 
    T = 1.0
    l = 201
    m = 16
    n= 64
    p=201
    pricer = spectral_collocation(option_type = "american_put",r=r,S=S,q=q,K=K,sigma =sigma,T=T,method = "default",collocation_num_n=n,quadrature_num_l =l,iteration_num_m=m,itegral_num_p=p)
    price =  pricer.solve()
    benchmark = price[0] - price[1] 
    print("Benchmark premium: ", benchmark)
    l_list = [5,7,11,15,15,25,25,25,35,51,65]
    m_list = [1,2,2,2,3,4,5,6,8,8,8]
    n_list = [4,5,5,6,7,9,12,15,16,24,32]
    p_list = [15,20,31,41,41,51,61,61,81,101,101]
    for m,l,n,p in zip(m_list,l_list,n_list,p_list):
        pricer = spectral_collocation(option_type = "american_put",r=r,S=S,q=q,K=K,sigma =sigma,T=T,method = "default",collocation_num_n=n,quadrature_num_l =l,iteration_num_m=m,itegral_num_p=p)
        price =  pricer.solve()
        p_list.append(price[0])
        print("(l,m,n) = ({},{},{}), p ={},American Premium = {},Relative Error: {},CPU Seconds:{} \n".format(l,m,n,p,price[0] - price[1], abs((price[0] - price[1] - benchmark)/benchmark)  ,price[2]))
        f = open('file.txt', 'a')
        f.write("(l,m,n) = ({},{},{}), p ={},American Premium = {},Relative Error: {},CPU Seconds:{} \n".format(l,m,n,p,price[0] - price[1],abs((price[0] - price[1] - benchmark)/benchmark)  ,price[2]))

        f.close()
     
 


   # print("european price = ", solver.european_price)
    #print("american price = ", price)

    # #make a plot for exercise boundary
    # plt.plot(solver.shared_tau, solver.shared_B, 'o-')
    # plt.plot(solver.shared_tau, solver.shared_B0, 'r--')
    # plt.legend(["real exercise boundary", "initial guess"])
    # plt.xlabel("Time to maturity tau")
    # plt.ylabel("Exercise boundary [$]")
    # plt.show()

    # plt.figure(2)
    # iters = np.array([float(x[0]) for x in solver.iter_records])
    # errors = np.array([x[1] for x in solver.iter_records])
    # plt.loglog(iters, errors, 'o-')
    # plt.xlabel("Number of iterations")
    # plt.ylabel("Match condition error")
    # plt.show()]
    #two things now: interp and itera which comes first 
    #parameters in table 2 
    # chebyshev 
 


    # process_num = 12 
    # process_list = []
    # S_list = list(range(80,121))
    # for i in range(12):
        
    #     code_list = S_list[i*3:(i+1)*3]
    #     if i == 11:
    #         code_list = S_list[i*3:]
        
        
    #     p = Process(target=single_delta, args=(code_list, i))
    #     p.start()
    #     process_list.append(p)
    #     print("Process {}".format(i))

    # for p in process_list:
        
    #     p.join()
 
 

    