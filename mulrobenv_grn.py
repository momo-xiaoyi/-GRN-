# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 23:04:38 2020

@author: ython
"""


import numpy as np
import matplotlib.pyplot as plt
import gym
from gym import spaces
from gym.utils import seeding

class robot:
    def __init__(self,n_r,n_o):
        self.xp=np.random.uniform(-1,1,n_r)#用均匀分布（0，1）初始化机器人位置
        self.yp=np.random.uniform(-1,1,n_r)
        self.xv=np.zeros(n_r)#用0初始化机器人速度
        self.yv=np.zeros(n_r)
        self.time=0.002#每次更新的步长/持续某速度的时间
        self.maxv=2./1.414#设定机器人最大速度为1
        self.obs=self.OBS(n_o)
        self.targ=self.targ()
        self.c_rang = 2*np.pi*1/n_r #通信范围的半径设置为1
        #self.s_s=2*np.pi*1/n_r#定义安全范围
        self.rep_w=np.zeros(n_r)#初始化排斥权重，排斥力等于排斥方向*权重
        self.nt=self.nearest_t(self.xp,self.yp,self.targ.xp,self.targ.yp)#初始化最近的目标
        self.no=self.nearest_o(self.xp,self.yp,self.obs.xp,self.obs.yp,self.c_rang)#初始化最近障碍物  
        self.grn_px=0
        self.grn_py=0
        self.a=44
        self.m=6
        self.c=70
        self.r=79
        self.b=420
        self.grn_pvx=np.zeros(n_r)#用0初始化基因调控网络速度
        self.grn_pvy=np.zeros(n_r)
    def update_p(self):
        self.update_v()
        self.xp=self.xp+self.xv*self.time
        self.yp=self.yp+self.yv*self.time
        self.grn_px=self.grn_px+self.grn_pvx*self.time
        self.grn_py=self.grn_py+self.grn_pvy*self.time
        
    def update_v(self):
        self.nt.update(self.xp,self.yp,self.targ.xp,self.targ.yp)#更新最近目标
        self.no.update(self.xp,self.yp,self.obs.xp,self.obs.yp)#更新最近的障碍物
        # x_at=-np.tanh(self.nt.xp - self.xp)*self.maxv#目标的吸引力为最近目标位置-自身,越近越小，正比例，不变
        # y_at=-np.tanh(self.nt.yp - self.yp)*self.maxv
        # #障碍物的排斥力为最近障碍物位置-自身，越近越大，反比例，在计算最近障碍物的时候已经变形了
        # x_rep=-np.tanh(self.no.xp - self.xp)*self.maxv 
        # y_rep=-np.tanh(self.no.yp - self.yp)*self.maxv
        x_at=-(self.nt.xp - self.xp)#目标的吸引力为最近目标位置-自身,越近越小，正比例，不变
        y_at=-(self.nt.yp - self.yp)
        #障碍物的排斥力为最近障碍物位置-自身，越近越大，反比例，在计算最近障碍物的时候已经变形了
        x_rep=-(self.no.xp - self.xp)/np.sqrt(np.power(self.no.xp-self.xp,2)+np.power(self.no.yp-self.yp,2))*self.no.num
        y_rep=-(self.no.yp - self.yp)/np.sqrt(np.power(self.no.xp-self.xp,2)+np.power(self.no.yp-self.yp,2))*self.no.num
        self.xv=-self.a*x_at+self.m*self.grn_px#更新速度
        self.yv=-self.a*y_at+self.m*self.grn_py
        self.grn_pvx=-self.c*self.grn_px+self.r*(1-np.exp(-x_at)) / (1 + np.exp(-x_at))+self.b*x_rep
        self.grn_pvy=-self.c*self.grn_py+self.r*(1-np.exp(-y_at)) / (1 + np.exp(-y_at))+self.b*y_rep
        

        
        
    class targ:
        def __init__(self):#初始化目标，100表示目标点精度
            t=np.linspace(-np.pi, np.pi, 100)#目标为单位圆
            self.xp=np.cos(t)
            self.yp=np.sin(t)
                
    class OBS:
        def __init__(self,n_o):
            self.xp=np.random.uniform(-1,1,n_o)#用均匀分布（0，1）初始化障碍物位置
            self.yp=np.random.uniform(-1,1,n_o)

    class nearest_t:#最近的目标
        def __init__(self,rob_x,rob_y,targ_x,targ_y):
            self.xp=np.array([])
            self.yp=np.array([])
            for index in np.arange(len(rob_x)):
                x=rob_x[index]-targ_x
                y=rob_y[index]-targ_y
                dis=np.sqrt(x**2+y**2)
                nt_id=np.argmin(dis)#机器人最近目标的id
                self.xp=np.append(self.xp, targ_x[nt_id])
                self.yp=np.append(self.yp, targ_y[nt_id])

        def update(self,rob_x,rob_y,targ_x,targ_y):#再次调用更新最近目标
            self.xp=np.array([])
            self.yp=np.array([])
            for index in np.arange(len(rob_x)):
                x=rob_x[index]-targ_x
                y=rob_y[index]-targ_y
                dis=np.sqrt(x**2+y**2)
                nt_id=np.argmin(dis)#机器人最近目标的id
                self.xp=np.append(self.xp, targ_x[nt_id])
                self.yp=np.append(self.yp, targ_y[nt_id])
              


class MulRobEnv(robot,gym.Env,):
    def __init__(self,n_r,n_o,n_c):
        self.n_o=n_o
        self.n_r=n_r
        self.n_c=n_c
        robot.__init__(self,n_r,n_o)#继承robot的初始化
        self.n_r=n_r
        #机器人和障碍物的位置为状态空间,状态空间为无穷
        high = np.array([np.ones(shape=(n_r+n_o))*np.inf,np.ones(shape=(n_r+n_o))*np.inf], dtype=np.float32)
        low = np.array([np.ones(shape=(n_r+n_o))*(-np.inf), np.ones(shape=(n_r+n_o))*(-np.inf)], dtype=np.float32)
        self.observation_space=spaces.Box(low=low, high=high, dtype=np.float32)
        #动作空间的最大值最小值
        high = np.array([np.ones(shape=(n_r))*self.maxv,np.ones(shape=(n_r))* self.maxv], dtype=np.float32)
        low = np.array([np.ones(shape=(n_r))*(-self.maxv), np.ones(shape=(n_r))*(-self.maxv)], dtype=np.float32)
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)#动作空间
        self.seed()

        
    #覆盖原来的nearest_o
    class nearest_o:
        def __init__(self,rob_x,rob_y,obs_x,obs_y,c_rang):
            self.xp=np.array([])
            self.yp=np.array([])
            self.num=np.array([])
            self.c_rang=c_rang
            obs_x_ =np.append(rob_x,obs_x)
            obs_y_ =np.append(rob_y,obs_y)
            for index in np.arange(len(rob_x)):
                x=rob_x[index]-obs_x_
                y=rob_y[index]-obs_y_
                dis=np.sqrt(x**2+y**2)
                nt_id=np.argmin(dis)#机器人最近障碍物的id，是自己，得找第二大的
                dis[nt_id]=np.inf#先把自己的距离定义为最大值
                in_obs_x=obs_x_[dis<self.c_rang]#求在通信范围内障碍物的坐标
                in_obs_y=obs_y_[dis<self.c_rang]
                #如果距离在安全范围内则不变，如果距离大于安全距离，如果大于，则当作距离为安全距离，来做变换
                if in_obs_x.size != 0:#通信范围有障碍物
                    self.xp=np.append(self.xp, np.mean(in_obs_x) )
                    self.yp=np.append(self.yp, np.mean(in_obs_y) )
                    self.num=np.append(self.num, in_obs_x.size )
                else:
                    theta=np.random.uniform()*2*np.pi
                    self.xp=np.append(self.xp,self.c_rang*np.cos(theta))
                    self.yp=np.append(self.yp,self.c_rang*np.sin(theta))
                    self.num=np.append(self.num, in_obs_x.size )
                
        def update(self,rob_x,rob_y,obs_x,obs_y):#覆盖其中的update
            self.xp=np.array([])
            self.yp=np.array([])
            self.num=np.array([])
            obs_x_ =np.append(rob_x,obs_x)
            obs_y_ =np.append(rob_y,obs_y)
            for index in np.arange(len(rob_x)):
                x=rob_x[index]-obs_x_
                y=rob_y[index]-obs_y_
                dis=np.sqrt(x**2+y**2)
                nt_id=np.argmin(dis)#机器人最近障碍物的id，是自己，得找第二大的
                dis[nt_id]=np.inf#先把自己的距离定义为最大值
                in_obs_x=obs_x_[dis<self.c_rang]#求在通信范围内障碍物的坐标
                in_obs_y=obs_y_[dis<self.c_rang]
                #如果距离在安全范围内则不变，如果距离大于安全距离，如果大于，则当作距离为安全距离，来做变换
                if in_obs_x.size != 0:#通信范围有障碍物
                    self.xp=np.append(self.xp, np.mean(in_obs_x) )
                    self.yp=np.append(self.yp, np.mean(in_obs_y) )
                    self.num=np.append(self.num, in_obs_x.size )
                else:
                    theta=np.random.uniform()*2*np.pi
                    self.xp=np.append(self.xp,self.c_rang*np.cos(theta))
                    self.yp=np.append(self.yp,self.c_rang*np.sin(theta))
                    self.num=np.append(self.num, in_obs_x.size )
                                      
    def updateBatch(self,rob_x,rob_y,obs_x,obs_y):#覆盖其中的update
        self.xpBatch=np.empty(shape=rob_x.shape)
        self.ypBatch=np.empty(shape=rob_y.shape)
        for jndex in range(rob_x.shape[0]):
            xp =np.array([])
            yp = np.array([])
            obs_x_ =np.append(rob_x[jndex],obs_x[jndex]) if obs_x.size != 0 else rob_x[jndex]
            obs_y_ =np.append(rob_y[jndex],obs_y[jndex]) if obs_y.size != 0 else rob_y[jndex]
            for index in np.arange(len(rob_x[jndex])):
                x=rob_x[jndex][index]-obs_x_
                y=rob_y[jndex][index]-obs_y_
                dis=np.sqrt(x**2+y**2)
                nt_id=np.argmin(dis)#机器人最近障碍物的id，是自己，得找第二大的
                dis[nt_id]=np.inf#先把自己的距离定义为最大值
                in_obs_x=obs_x_[dis<self.c_rang]#求在通信范围内障碍物的坐标
                in_obs_y=obs_y_[dis<self.c_rang]
                #如果距离在安全范围内则不变，如果距离大于安全距离，如果大于，则当作距离为安全距离，来做变换
                if dis[nt_id] < self.c_rang:
                    xp=np.append(xp, np.mean(in_obs_x) )
                    yp=np.append(yp, np.mean(in_obs_y) )
                else:
                    theta=np.random.uniform()*2*np.pi
                    xp=np.append(xp,self.c_rang*np.cos(theta))
                    yp=np.append(yp,self.c_rang*np.sin(theta))
            self.xpBatch[jndex]=xp
            self.ypBatch[jndex]=yp
    
    def updatesc(self,rob_x,rob_y,obs_x,obs_y,c_rang,n_c):
        n_r = len(rob_x)
        self.sc=np.empty(shape=(n_r,n_c))
        obs_x_ =np.append(rob_x,obs_x)
        obs_y_ =np.append(rob_y,obs_y)
        for index in np.arange(len(rob_x)):
            #与最近目标距离
            x=rob_x[index]-self.targ.xp
            y=rob_y[index]-self.targ.yp
            dis=np.sqrt(x**2+y**2)
            self.sc[index,0]=np.min(dis)#机器人最近目标的id
            
            #与最近障碍物距离
            x=rob_x[index]-obs_x_
            y=rob_y[index]-obs_y_
            dis=np.sqrt(x**2+y**2)
            nt_id=np.argmin(dis)#机器人最近障碍物的id，是自己，得找第二大的
            dis[nt_id]=np.inf#先把自己的距离定义为最大
            self.sc[index,1]=np.min(dis) if np.min(dis)<self.c_rang else self.c_rang
        return self.sc

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    
    def reset(self):#初始化观测
        self.xp=np.random.uniform(-1,1,self.n_r)#用均匀（0，1）初始化机器人位置
        self.yp=np.random.uniform(-1,1,self.n_r)
        self.obs=self.OBS(self.n_o)
        state_x = np.append(self.xp, self.obs.xp)
        state_y = np.append(self.yp, self.obs.yp)
        self.state = np.array([state_x,state_y])
        
        self.grn_px=0
        self.grn_py=0
        return self._get_obser()
    
    def _get_obser(self):#获得观测值
        state_x,state_y = self.state
        
        return np.array([state_x,state_y])
    
    def step(self, pre):#输入动作，得到s_, r, done, info,这里还得更新状态
        self.nt.update(self.xp,self.yp,self.targ.xp,self.targ.yp)#更新最近目标
        self.no.update(self.xp,self.yp,self.obs.xp,self.obs.yp)#更新最近的障碍物
        # x_at = self.nt.xp - self.xp
        # y_at = self.nt.yp - self.yp
        # attra = np.array([x_at, y_at])
        # action = attra - pre.T   
        # self.xv = action[0,:]
        # self.yv = action[1,:]
        # #更新位置
        # self.xp=self.xp+self.xv*self.time
        # self.yp=self.yp+self.yv*self.time
        self.update_p()
        #更新状态
        self.nt.update(self.xp,self.yp,self.targ.xp,self.targ.yp)#更新最近目标
        self.no.update(self.xp,self.yp,self.obs.xp,self.obs.yp)#更新最近的障碍物

        
        # #相对位置的极坐标
        thetadot = np.sqrt(np.power(self.no.xp-self.xp,2)+np.power(self.no.yp-self.yp,2))
        cost_pre = np.log(thetadot/self.c_rang)
        cost_pre = cost_pre
        thetadot = - np.sqrt(np.power(self.nt.xp-self.xp,2)+np.power(self.nt.yp-self.yp,2))
        cost_tar2 = thetadot/self.c_rang
        cost = cost_tar2 + cost_pre
        return self._get_obser(), cost, False, {}
    
if __name__ == '__main__':

    n_r=20
    n_o=0
    n_c=1+1#价值网络处理的量
    #环境###########
    env = MulRobEnv(n_r,n_o,n_c)
    
    plt.ion()
    fig, ax=plt.subplots(1)
    plt.xlim((-5, 5))
    plt.ylim((-5, 5))
    rob_anim, = ax.plot([], [],'.')
    obs_anim, = ax.plot([],[],'.')

    state = env.reset()
    state = state.T
    for j in range(400):
        rob_anim.set_xdata(env.xp)
        rob_anim.set_ydata(env.yp)
        obs_anim.set_xdata(env.obs.xp)
        obs_anim.set_ydata(env.obs.yp)
        if j==0:
            plt.pause(3)
        plt.draw()
        plt.pause(0.001)
        action=np.array([])
        state_, reward, done, _ = env.step(action)
        state_ =state_.T
        state = state_


        


