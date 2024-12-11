# -*- coding: utf-8 -*-
import numpy as np
import math

class GameTest:
    def __init__(self, P_dic, coef):
        self.n = P_dic['n']
        self.cn = P_dic['cn']     
        self.company_strategies = np.zeros(self.cn)
        self.platform_strategies = np.array([P_dic['a'],P_dic['b']])
        self.pd = np.array(self.adjust(coef))


    def payoff_platform(self):
        b = sum(self.company_strategies) + sum([math.log(v,k+1) for k,v in zip([1,2,3],self.pd*self.n)])
        c = sum(self.platform_strategies) * math.log(self.n, 10)
        return b-c
    
    def payoff_company(self):
        boef = sum([(3-k)*math.log(v,k+1) for k,v in zip([1,2,3],self.pd*self.n)]) * (math.log(self.platform_strategies[0], 10) + 1/self.cn) 
        b = np.array([boef*r/sum(self.company_strategies) for r in self.company_strategies])
        c = self.company_strategies
        return b-c


    def random_payoff_result(self):
        sum = 1000
        payoff_platform = 0.0
        payoff_company = np.zeros(5)
        for i in range (sum):
            self.company_strategies = np.random.choice(a=np.arange(1,10), size = self.cn)
            payoff_platform += self.payoff_platform()
            payoff_company += self.payoff_company()
        ap = payoff_platform / sum
        ac = payoff_company / sum
        return {'rnd_platform': ap, 'rnd_company': ac}


    def average_payoff_result(self):
        payoff_platform = 0.0
        payoff_company = np.zeros(5)
        for i1 in range(1,10):
            for i2 in range(1,10):
                for i3 in range(1,10):
                    for i4 in range(1,10):
                        for i5 in range(1,10):
                            self.company_strategies = np.array([i1,i2,i3,i4,i5])
                            payoff_platform += self.payoff_platform()
                            payoff_company += self.payoff_company()
        sum = math.pow(9,5)
        ap = payoff_platform / sum
        ac = payoff_company / sum
        return {'ave_platform': ap, 'ave_company': ac}
    
    def adjust(self, payoff):
        tp = np.array(payoff)
        if (tp <= 0).any():
            tp[tp<=0] = 0.01
        return [ r/sum(tp) for r in tp]
      


if __name__ == '__main__':
    print_hi('Something Wrong!')

