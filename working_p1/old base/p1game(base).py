# -*- coding: utf-8 -*-
import numpy as np

class GameS1:
    def __init__(self, F_dic, coef):
        self.fdic = F_dic
        self.n = len(F_dic)
        self.ids = {}
        self.user_strategies = np.random.choice(a=np.arange(1, 4), size = self.n, p=coef)
        self.pd = np.unique(self.user_strategies, return_counts=True)
        self.setIDs()

    def setIDs(self):  
        count = 0
        for key in self.fdic:
            self.ids[int(key)] = count
            count+=1
        print(count)

    def run_slot(self, k):
        #for i in k:
            #key = np.random.choice(list(self.fdic.keys()))
        for key in self.fdic:
            friends = np.fromstring(self.fdic[key], dtype=int, sep=' ')
            count_1 = 0
            count_2 = 0
            count_3 = 0
            for f in friends:
                temp = self.user_strategies[self.ids[f]]
                if temp==1:
                    count_1+=1
                elif temp==2:
                    count_2+=1
                elif temp==3:
                    count_3+=1
                else:
                    raise Exception("strategy get error!")
            max_count = max(count_1,count_2,count_3)
            if count_1 == max_count:
                self.user_strategies[self.ids[int(key)]]=1
            elif count_2 == max_count:
                self.user_strategies[self.ids[int(key)]]=2
            elif count_3 == max_count:
                self.user_strategies[self.ids[int(key)]]=3
            else:
                raise Exception("strategy set error!")
        self.pd = np.unique(self.user_strategies, return_counts=True)


class GameS2:
    def __init__(self, F_dic, P_dic, coef):
        self.fdic = F_dic
        self.n = len(F_dic)
        self.ids = {}
        self.user_strategies = np.random.choice(a=np.arange(1, 4), size = self.n, p=coef)
        self.pd = np.unique(self.user_strategies, return_counts=True)
        self.setIDs()
        self.setPmatrix(P_dic)

    def setIDs(self):  
        count = 0
        for key in self.fdic:
            self.ids[int(key)] = count
            count+=1
        print(count)
    
    def setPmatrix(self, P_dic):
        la = P_dic['la']
        si = P_dic['si']
        self.pmatrix = np.array([
            (0,0,la),
            (si-la,0,la),
            (si,si,0)
        ])
        self.wo = P_dic['wo']

    def run_slot(self, k):
        #for i in k:
            #key = np.random.choice(list(self.fdic.keys()))
        for key in self.fdic:
            friends = np.fromstring(self.fdic[key], dtype=int, sep=' ')
            payoff_1 = 0
            payoff_2 = 0
            payoff_3 = 0
            for f in friends:
                flabel = self.user_strategies[self.ids[f]]
                if flabel==1:
                    payoff_1+=self.compute_payoff(f, 1)
                elif flabel==2:
                    payoff_2+=self.compute_payoff(f, 2)
                elif flabel==3:
                    payoff_3+=self.compute_payoff(f, 3)
                else:
                    raise Exception("strategy get error!")
            payoff = payoff_1+payoff_2+payoff_3
            self.user_strategies[self.ids[int(key)]]=np.random.choice(a=np.arange(1, 4), p=[payoff_1/payoff,payoff_2/payoff,payoff_3/payoff])
        self.pd = np.unique(self.user_strategies, return_counts=True)
    
    def compute_payoff(self, fkey, flabel):
        payoff = 0
        fif = np.fromstring(self.fdic[str(fkey)], dtype=int, sep=' ')
        for fi in fif:
            tlabel = self.user_strategies[self.ids[fi]]
            payoff += self.pmatrix[flabel-1][tlabel-1]
        return 1-self.wo+self.wo*payoff


class GameS3:
    def __init__(self, F_dic, P_dic, coef):
        self.fdic = F_dic
        self.n = len(F_dic)
        self.ids = {}
        self.user_strategies = np.random.choice(a=np.arange(1, 4), size = self.n, p=coef)
        self.pd = np.unique(self.user_strategies, return_counts=True)
        self.setIDs()
        self.setPmatrix(P_dic)

    def setIDs(self):  
        count = 0
        for key in self.fdic:
            self.ids[int(key)] = count
            count+=1
        print(count)
    
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

    def run_slot(self, k):
        keys = np.random.choice(a=list(self.fdic.keys()), size=k)
        for key in keys:
            friends = np.fromstring(self.fdic[key], dtype=int, sep=' ')
            payoff_1 = 0
            payoff_2 = 0
            payoff_3 = 0
            for f in friends:
                flabel = self.user_strategies[self.ids[f]]
                if flabel==1:
                    payoff_1+=(1-self.wo)*self.G1+self.wo*self.compute_payoff(f, 1)
                elif flabel==2:
                    payoff_2+=(1-self.wo)*self.G2+self.wo*self.compute_payoff(f, 2)
                elif flabel==3:
                    payoff_3+=(1-self.wo)*self.G3+self.wo*self.compute_payoff(f, 3)
                else:
                    raise Exception("strategy get error!")
            payoff = payoff_1+payoff_2+payoff_3
            self.user_strategies[self.ids[int(key)]]=np.random.choice(a=np.arange(1, 4), p=[payoff_1/payoff,payoff_2/payoff,payoff_3/payoff])
        self.pd = np.unique(self.user_strategies, return_counts=True)
    
    def compute_payoff(self, fkey, flabel):
        payoff = 0
        fif = np.fromstring(self.fdic[str(fkey)], dtype=int, sep=' ')
        for fi in fif:
            tlabel = self.user_strategies[self.ids[fi]]
            payoff += self.pmatrix[flabel-1][tlabel-1]
        return payoff


if __name__ == '__main__':
    print_hi('Something Wrong!')

