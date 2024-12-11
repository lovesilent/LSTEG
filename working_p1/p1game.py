# -*- coding: utf-8 -*-
import numpy as np
import math

class GameM:
    def __init__(self, F_dic, P_dic, coef):
        self.n = len(F_dic)
        self.ids = np.ones(self.n, dtype=int)
        self.ne_payoffs = np.ones(self.n, dtype=int)
        self.fdic = []
        self.user_strategies = np.random.choice(a=np.arange(1, 4), size = self.n, p=coef)
        self.pd = np.unique(self.user_strategies, return_counts=True)
        self.setIDs(F_dic)
        self.transFdic(F_dic)
        self.setPmatrix(P_dic)

    def setIDs(self, F_dic):  
        count = 0
        for key in F_dic:
            self.ids[count] = int(key)
            count+=1
        print(count)


    def transFdic(self, F_dic):
        for key in F_dic:
            friends = np.fromstring(F_dic[key], dtype=int, sep=' ')
            fnumber = np.size(friends)
            fnp = np.zeros(fnumber, dtype=int)
            for x in range(fnumber):
                fnp[x] =  np.where(self.ids == friends[x])[0]
            self.fdic.append(fnp)

    
    def setPmatrix(self, P_dic):
        la = P_dic['la']
        si = P_dic['si']
        self.pmatrix = np.array([
            (0,0,la),
            (si-la,0,la),
            (si,si,0)
        ])
        self.wo = P_dic['wo']
        self.G1 = P_dic['G1']
        self.G2 = P_dic['G2']
        self.G3 = P_dic['G3']

    def run_slot(self):
        self.set_ne_payoffs()
        for id in range(self.n):
            friends = self.fdic[id]
            payoff_1 = 0
            payoff_2 = 0
            payoff_3 = 0
            for fid in friends:
                flabel = self.user_strategies[fid]
                if flabel==1:
                    payoff_1+=(1-self.wo)*self.G1+self.wo*self.ne_payoffs[fid]
                elif flabel==2:
                    payoff_2+=(1-self.wo)*self.G2+self.wo*self.ne_payoffs[fid]
                elif flabel==3:
                    payoff_3+=(1-self.wo)*self.G3+self.wo*self.ne_payoffs[fid]
                else:
                    raise Exception("strategy get error!")
            self.user_strategies[id]=np.random.choice(a=np.arange(1, 4), p=self.adjust([payoff_1, payoff_2, payoff_3]))
        self.pd = np.unique(self.user_strategies, return_counts=True)
    

    def set_ne_payoffs(self):
        self.ne_payoffs = np.zeros(self.n)
        for id in range(self.n):
            label = self.user_strategies[id]
            self.ne_payoffs[id] = self.compute_payoff(id, label)


    def compute_payoff(self, id, label):
        payoff = 0
        friends = self.fdic[id]
        for fid in friends:
            tlabel = self.user_strategies[fid]
            payoff += self.pmatrix[label-1][tlabel-1]
        return payoff

    
    def adjust(self, payoff):
        tp = np.array(payoff)
        if (tp <= 0).any():
            tp[tp<=0] = 0.01
        return [ r/sum(tp) for r in tp]
            

class GameB:
    def __init__(self, F_dic, P_dic, coef, tag):
        self.n = len(F_dic)
        self.ids = np.ones(self.n, dtype=int)
        self.ne_payoffs = np.ones(self.n, dtype=int)
        self.fdic = []
        self.user_strategies = np.random.choice(a=np.arange(1, 4), size = self.n, p=coef)
        self.company_strategies = np.random.choice(a=np.arange(1,10), size = P_dic['cn'])
        self.platform_strategies = np.array([P_dic['a'],P_dic['b']])
        self.pd = np.unique(self.user_strategies, return_counts=True)
        self.setIDs(F_dic)
        if tag is None:
            self.transFdic(F_dic)
        else:
            self.fdic = tag
        self.setPmatrix(P_dic)


    def setIDs(self, F_dic):  
        count = 0
        for key in F_dic:
            self.ids[count] = int(key)
            count+=1
        print(count)


    def transFdic(self, F_dic):
        for key in F_dic:
            friends = np.fromstring(F_dic[key], dtype=int, sep=' ')
            fnumber = np.size(friends)
            fnp = np.zeros(fnumber, dtype=int)
            for x in range(fnumber):
                fnp[x] =  np.where(self.ids == friends[x])[0]
            self.fdic.append(fnp)

    
    def setPmatrix(self, P_dic):
        la = P_dic['la']
        si = P_dic['si']
        self.pmatrix = np.array([
            (0,0,la),
            (si-la,0,la),
            (si,si,0)
        ])
        self.wo = P_dic['wo']
        self.G1 = math.log(self.platform_strategies[0],10)
        self.G2 = (math.log(self.platform_strategies[0],10) + math.log(self.platform_strategies[1],10))/2
        self.G3 = math.log(self.platform_strategies[1],10)

    def payoff_platform(self):
        b = sum(self.company_strategies) + sum([math.log(v,k+1) for k,v in zip(self.pd[0],self.pd[1])])
        c = sum(self.platform_strategies) * math.log(self.n, 10)
        return b-c
    
    def payoff_company(self):
        boef = sum([(3-k)*math.log(v,k+1) for k,v in zip(self.pd[0],self.pd[1])]) * (math.log(self.platform_strategies[0],10) + 1)
        c = self.company_strategies
        b = np.array([boef*r/sum(c) for r in c])
        return b-c

    

    def run_slot(self):
        self.set_ne_payoffs()
        for id in range(self.n):
            friends = self.fdic[id]
            payoff_1 = 0
            payoff_2 = 0
            payoff_3 = 0
            for fid in friends:
                flabel = self.user_strategies[fid]
                if flabel==1:
                    payoff_1+=(1-self.wo)*self.G1+self.wo*self.ne_payoffs[fid]
                elif flabel==2:
                    payoff_2+=(1-self.wo)*self.G2+self.wo*self.ne_payoffs[fid]
                elif flabel==3:
                    payoff_3+=(1-self.wo)*self.G3+self.wo*self.ne_payoffs[fid]
                else:
                    raise Exception("strategy get error!")
            self.user_strategies[id]=np.random.choice(a=np.arange(1, 4), p=self.adjust([payoff_1, payoff_2, payoff_3]))
        self.pd = np.unique(self.user_strategies, return_counts=True)
    

    def set_ne_payoffs(self):
        self.ne_payoffs = np.zeros(self.n)
        for id in range(self.n):
            label = self.user_strategies[id]
            self.ne_payoffs[id] = self.compute_payoff(id, label)


    def compute_payoff(self, id, label):
        payoff = 0
        friends = self.fdic[id]
        for fid in friends:
            tlabel = self.user_strategies[fid]
            payoff += self.pmatrix[label-1][tlabel-1]
        return payoff

    
    def adjust(self, payoff):
        tp = np.array(payoff)
        if (tp <= 0).any():
            tp[tp<=0] = 0.01
        return [ r/sum(tp) for r in tp]
        


if __name__ == '__main__':
    print_hi('Something Wrong!')

