import random
from scipy.fft import fft, fftfreq
from scipy.signal import hilbert, find_peaks
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, freqz, sosfilt, buttord, firwin
import copy

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs  # 奈奎斯特频率
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='bandpass', output = 'sos')
    return sos

# 应用滤波器
# def bandpass_filter(data, lowcut, highcut, fs, order=5):
#     sos = butter_bandpass(lowcut, highcut, fs, order=order)
#     y = sosfilt(sos, data)  # 使用filtfilt实现零相位滤波
#     return y

def butter_lowpass(lowcut, fs, order=5):
    nyq = 0.5 * fs  # 奈奎斯特频率
    low = lowcut / nyq
    sos = butter(order, low, btype='lowpass', output = 'sos')
    return sos

# 应用滤波器
def lowpass_filter(data, lowcut, fs, order=5):
    # sos = butter_lowpass(lowcut, fs, order=order)
    # y = sosfilt(sos, data)  # 使用filtfilt实现零相位滤波
    taps = design_fir_lowpass(lowcut,fs, numtaps = 101)
    y = filtfilt(taps, [1.0], data)  # 使用filtfilt实现零相位滤波
    return y
    
def design_fir_bandpass(lowcut, highcut, fs, numtaps=1001):
    nyq = 0.5 * fs
    taps = firwin(numtaps, [lowcut, highcut], pass_zero=False, fs=fs)
    return taps

def design_fir_lowpass(highcut, fs, numtaps=101):
    nyq = 0.5 * fs
    taps = firwin(numtaps, highcut, pass_zero='lowpass', fs=fs, window = 'hamming')
    return taps

def bandpass_filter(data, lowcut, highcut, fs):
    taps = design_fir_bandpass(lowcut, highcut,fs, numtaps = 301)
    y = filtfilt(taps, [1.0], data)  # 使用filtfilt实现零相位滤波
    return y



def Interp_Mean(f, t_idx, t_Lower, t_Upper):
    import numpy as np
    idx_inrange = np.where((t_idx>t_Lower) & (t_idx<t_Upper))[0]
    t_supersample = np.linspace(t_Lower, t_Upper, len(idx_inrange)*10)
    result = np.mean(f(t_supersample))
    return result

def OMP(y, library,scale, sparse_threshold = 1e-3 ,stop_tolerance = 1e-4, step_tolerance = 1e-5, sparse_max = 6,smooth_window = 1,w_A2b = 1, fixed = None):
    N,M,feature_length = np.shape(library)
    coef = np.zeros(feature_length)
    y_hat = np.copy(y)
    y_norm = np.max(np.abs(y_hat),axis = 1)
    if fixed == None:
        
        S = []
    else:
        S = fixed
    r0 = 0
    for i in range(M):
        r0 += np.sum(y_hat[:,i]**2/N)
    coef_new = Xi(y,library,S)
    r = 0
    for i in range(M):
        r += np.sum(((y[:,i] - np.dot(library[:,i,:], coef_new)))**2/N)
        y_hat[:,i] = y[:,i] - np.dot(library[:,i,:], coef_new)

    loss = [1]
    for n in range(sparse_max):
        r_long = y_hat.reshape(-1)
        Cdots = np.full(feature_length, -np.inf)  # 选中过的设为 -inf
        for i in range(feature_length):
            if i in S:
                continue
            r_try = 0
            S_try = S+[i]
            Xi_try = Xi(y,library,S_try)
            for j in range(M):
                r_try += np.sum(((y[:,j] - np.dot(library[:,j,:], Xi_try)))**2/N)
            Cdots[i] = -r_try
        
        Idx = np.argmax(Cdots)
        S.append(Idx)
        print(Cdots)
        print(S)
        coef_new = Xi(y,library,S)

        r_new = 0
        for i in range(M):
            r_new += np.sum(((y[:,i] - np.dot(library[:,i,:], coef_new)))**2/N)
            y_hat[:,i] = y[:,i] - np.dot(library[:,i,:], coef_new)

        loss.append((r_new)/r0)
        print(r_new/r0)
        print((r-r_new)/r0)
        coef = coef_new
        if (r_new)/r0<stop_tolerance or (r-r_new)/r0 < step_tolerance:  
            break
        r = r_new

    contributions = np.zeros((feature_length,M))

    for i in range(feature_length):
        for m in range(M):
            phi_i_m = coef[i] * library[:,m,i]
            contributions[i,m] = np.sum(phi_i_m**2/y[:,m]**2)/N

    print(contributions)
    for m in range(M):
        contributions[:,m] = contributions[:,m]/np.max(contributions[:,m])

    contributions = np.sum(contributions, axis = 1)

    # print(contributions)
    for i in range(feature_length):
        if (contributions[i] < sparse_threshold):
            if i in S:
                S.remove(i)
    
    coef = Xi(y,library,S)


    return coef



def cyxpy_solver(y, library,order, lam = 1e-3 ,stop_tolerance = 1e-4, step_tolerance = 1e-5, sparse_max = 6,smooth_window = 1,w_A2b = 1, fixed = None):
    import cvxpy as cp
    p = np.shape(library)[2]
    full_order = np.shape(library)[1]
    
    c = cp.Variable(p)
    loss = 0
    X_k = []
    y_k = []
    y_scale = np.ones(len(order))
    for i in range(len(order)):
        X_k.append(library[:,order[i],:])
        y_k.append(y[:,order[i]])
        y_scale[i]=np.linalg.norm(y_k[i])
    for i in range(len(order)):
        loss+=cp.sum_squares(y_k[i] - X_k[i]@c)/y_scale[i] 
    for i in range(full_order):
        if i not in order:
            loss+=cp.sum_squares(library[:,i,:]@c) 
    obj = loss + lam * cp.norm1(c)
    prob = cp.Problem(cp.Minimize(obj))
    prob.solve(solver=cp.SCS)
    c_std = c.value
    score = np.sum(np.linalg.norm(c_std*X_k, axis = 1).T/y_scale,axis = 1)
    sorted_score = np.argsort(np.abs(score))[::-1]
    
    r = np.linalg.norm(y[:,order])
    r0 = np.linalg.norm(y[:,order])
    print('start:%.5f'%r)
    for i in range(min(len(sorted_score),sparse_max)):
        selected = list(sorted_score[:i+1])
        y_hat = np.zeros_like(y[:,order])
        c_iter = Xi(y[:,order],library[:,order,:],selected)
        for j in range(len(order)):
            if i == 0:
                y_hat[:,j] = (c_iter[selected]*library[:,order[j],selected]).reshape(-1)
            else:
                y_hat[:,j] = np.sum(c_iter[selected]*library[:,order[j],selected],axis = 1)
        r_new = np.linalg.norm(y_hat-y[:,order])
        # print(r_new)
        print('Round %d, step: %.5f, now: %.5f, selected: %d'%(i,(r-r_new)/r0,r_new/r0,selected[i]))
        # print(r_new/r0)
        # print((r-r_new)/r0)
        if (r_new)/r0<stop_tolerance:  #or (r-r_new)/r0 < step_tolerance:  
            break
        r = r_new
    coef = Xi(y[:,order],library[:,order,:],selected)


    return coef
    
        





def Xi(dot,library, S,Lambda = 1, alpha = 0):
    # 
    N,M,feature_length = np.shape(library)
    X = np.copy(library)
    y = np.copy(dot)
    scale = np.ones(M)
    for i in range(M):
        scale[i] = (np.max(y[:,i])- np.min(y[:,i]))
    # print(scale)
    # coef = np.zeros(feature_length)
    for i in range(M):
        y[:,i] = y[:,i]/scale[i]
        X[:,i,:] = X[:,i,:]/scale[i]
    X_S = np.zeros([N,M,len(S)])
    S_idx = 0
    Scale_X = np.zeros(len(S))
    for i in S:
        Scale_X[S_idx] = np.max(X[:,:,i]) - np.min(X[:,:,i])
        S_idx += 1
    # print(Scale_X)
    S_idx = 0
    for i in range(feature_length):
        if i in S:
            for n in range(M):
                X_S[:,n,S_idx] = X[:,n,i]/Scale_X[S_idx]
            # print(X_S[:,n,:])
            S_idx += 1
            
    numerator_1 = 0
    numerator_2 = 0
    Lambdas = np.ones(M)
    if Lambda != 1:
        Lambdas = Lambda

    
    for n in range(M):
        # print(np.linalg.cond(X_S[:,n,:]))
        numerator_1 += np.dot(X_S[:,n,:].T,X_S[:,n,:])*Lambdas[n]
        numerator_2 += np.dot(X_S[:,n,:].T, y[:,n])*Lambdas[n]
    # print(np.linalg.cond(numerator_1))
    # print(numerator_1)
    coef = np.dot(np.linalg.inv(numerator_1)+0*np.eye(len(S)), numerator_2)
    # coef = np.dot(np.linalg.pinv(np.dot(X[:,0,:].T,X[:,0,:]) + np.dot(X[:,1,:].T,X[:,1,:]) + alpha*np.eye(feature_length)), (np.dot(X[:,0,:].T, y[:,0]) +np.dot(X[:,1,:].T, y[:,1])))
    result = np.zeros(feature_length)
    S_idx = 0
    for i in range(feature_length):
        if i in S:
            result[i] = coef[S_idx]/Scale_X[S_idx]
            S_idx += 1
    
    return result

def corr(a,b):
    a_range = np.max(a) - np.min(a)
    b_range = np.max(b) - np.min(b)
    cov = np.sum(a/a_range*b/b_range)
    a_norm = np.linalg.norm(a/a_range, ord = 2)
    b_norm = np.linalg.norm(b/b_range, ord = 2)
    # print(cov)
    # print()
    return np.abs(cov)/(a_norm*b_norm)


def Get_Drift(x,Fs,base_frequency):
    low_cut = base_frequency/5
    a0 = lowpass_filter(x, low_cut, Fs)
    return a0

def Get_filterorder(wp,ws,gpass,gstop,fs):
    N, Wn = buttord(wp, ws, gpass, gstop, fs=3)
    return N

def Get_Harmonic(x,x0,order,Fs,base_frequency):
    low_cut = order*base_frequency-base_frequency/5
    high_cut = order*base_frequency+base_frequency/5
    low_stop = order*base_frequency-base_frequency/3
    high_stop = order*base_frequency+base_frequency/3
    harmonic_order = bandpass_filter(x,low_cut,high_cut,Fs)
    
    # x0 = np.sin(frequency*t)
    analytic_x = hilbert(harmonic_order)
    analytic_x0 = hilbert(x0)
    instantaneous_amplitude = np.abs(analytic_x)
    instantaneous_phase = np.unwrap(np.angle(analytic_x) - np.angle(analytic_x0))
    # b_cos = instantaneous_amplitude*np.sin(instantaneous_phase)
    # c_sin = instantaneous_amplitude*np.cos(instantaneous_phase)
    return instantaneous_amplitude, instantaneous_phase

class WeakNOForce:
    def __init__(self, x_dims,library,library_name, norm = True, zerobench = 1e-8, corrbench = 0.99, fixed = []):
        self.dims = x_dims # Degree of Freedom of weakly nonlinear oscillator
        self.evolutions = None
        self.t_evolutions = None
        self.frequencys = np.zeros(x_dims)
        self.library = library
        self.library_name = library_name
        self.data = None
        self.t = None
        self.length = None
        self.Phi = None
        self.Xi = np.zeros([self.dims,len(self.library)+1])
        self.predict = None
        self.norm = norm
        self.zerobench = zerobench
        self.corrbench = corrbench
        self.fixterms = []
        for i in range(self.dims):
            self.fixterms.append(None)
    
    def fixed(self,l,dims):
        self.fixterms[dims] = l

    # def __str__(self):
    #     s = ""
    #     varble = []
    #     for d in range(self.dims):
    #         varble.append("x%d"%d)
    #         varble.append("x%d'"%d)
    #     for d in range(self.dims):
    #         s_sub = "x%d'' + "%d
    #         for i in range(len(self.Xi[d])):
    #             if i == 0:
    #                 xi = self.frequencys[d]**2+self.Xi[d][i]
    #                 s_sub += "%e%s + "%(xi,self.library_name[i](varble))
    #             elif np.abs(self.Xi[d][i])>1e-12:
    #                 s_sub += "%e%s + "%(self.Xi[d][i],self.library_name[i](varble))
    #         s_sub = s_sub[:-3]
    #         s_sub += ' = 0\n'
    #         s+=s_sub
    #     return s    

    def Get_frequency(self,X,X_dot,t):
        from scipy.interpolate import interp1d
        # X: [length*dims]
        L = len(t)
        self.scale = np.ones(len(self.library))
        self.scale_base = np.ones(self.dims*2)
        # if self.norm:
        #     for i in range(self.dims):
        #         scale_i = 2*np.pi/np.max(X[:,i])
        #         X[:,i] = X[:,i]#*scale_i
        #         for j in range(2):
        #             self.scale_base[i*2+j] = scale_i
        #     for i in range(len(self.scale)):
        #         self.scale[i] = self.library[i](self.scale_base)
            
        self.data = X
        self.velocity = X_dot
        self.t = t
        N = np.power(2, np.ceil(np.log2(L)))
        Fs = 1/(t[1] - t[0])
        self.trend_f = []
        self.amplitude_f = []
        self.phase_f = []
        self.data_f = []
        self.velocity_f = []
        scale = np.ones(self.dims*2)
        for i in range(self.dims):
            scale[2*i] = (np.max(self.data[:,i]) - np.min(self.data[:,i]))/2
            scale[2*i+1] = (np.max(self.data[:,i]) - np.min(self.data[:,i]))/2
        self.scale = scale
        for i in range(self.dims):
            fft_x = np.abs(fft(self.data[:,i],n = int(N)))[range(int(N/2))]
            Freq = np.arange(int(N/2))*Fs/N
            # self.data[:,i] = self.data[:,i]/scale[2*i]
            # self.velocity[:,i] = self.velocity[:,i]/scale[2*i]
            
            self.frequencys[i] = Freq[np.argmax(fft_x[1:])]*2*np.pi
            self.data_f.append(interp1d(t,self.data[:,i], kind = "linear"))
            self.velocity_f.append(interp1d(t,self.velocity[:,i], kind = "linear"))    
        self.library_scale = np.ones(len(self.library)+1)
        for i in range(len(self.library)):
            self.library_scale[i] = self.library[i](self.scale)
            
            
        
        

    def Get_Evolution(self,order = [0,1],lowcut = None):
        from scipy.interpolate import interp1d
        # from scipy.interpolate import make_smoothing_spline
        from scipy.optimize import curve_fit
        self.dot = []
        self.evolutions = []
        self.evolutions_f = []
        periods_discrete_supersample = 10000
        dt = self.t[1]-self.t[0]
        self.order = order
        self.harmonic_number = 0
        self.harmonic_order = {}
        self.omegas = []
        
        for i in range(len(order)):
            if order[i] == 0:
                self.harmonic_order[order[i]] = [self.harmonic_number]
                self.harmonic_number+=1
            else:
                self.harmonic_order[order[i]] = []
                for j in range(2):
                    self.harmonic_order[order[i]].append(self.harmonic_number)
                    self.harmonic_number+=1


            
        for i in range(self.dims):
            omega_i = self.frequencys[i]
            base_frequency = omega_i/(2*np.pi)
            dt = self.t[1]-self.t[0]
            Fs = 1/dt
            T = 2*np.pi/omega_i
            num_period = int((self.t[-1] - self.t[0]) // T)
            t_periods = np.linspace(T,T*num_period, num_period+1)
            t_real = np.linspace(0,self.t[-1],int(len(self.t)/(T/dt)),endpoint = False)
            self.dot.append(np.zeros([num_period, self.harmonic_number]))
            self.evolutions.append(np.zeros([num_period,self.harmonic_number]))
            self.evolutions_f.append([])
            self.omegas.append(np.zeros(len(order)))
            for m in self.order:
                if m==0:# drift
                    instantaneous_drift_i = Get_Drift(self.data[:,i], Fs, base_frequency)
                    instantaneous_drift_f_i = interp1d(self.t, instantaneous_drift_i, kind = "linear", fill_value='extrapolate')
                    for n in range(num_period):
                        self.evolutions[i][n,m] = Interp_Mean(instantaneous_drift_f_i, self.t, t_periods[n], t_periods[n+1])
                    # self.evolutions_f[i].append(instantaneous_drift_f_i)
                else:
                    x0 = np.sin(omega_i*m*self.t)
                    A_i, P_i = Get_Harmonic(self.data[:,i],x0,m,Fs,base_frequency)
                    A_f_i = interp1d(self.t, A_i, kind = "linear", fill_value='extrapolate')
                    P_f_i = interp1d(self.t, P_i, kind = "linear", fill_value='extrapolate')
                    b_idx = self.harmonic_order[m][0]
                    c_idx = self.harmonic_order[m][1]
                    for n in range(num_period):
                        amplitude_i_n = Interp_Mean(A_f_i, self.t, t_periods[n], t_periods[n+1])
                        phase_i_n = Interp_Mean(P_f_i, self.t, t_periods[n], t_periods[n+1])
                        b_n = amplitude_i_n*np.sin(phase_i_n)
                        c_n = amplitude_i_n*np.cos(phase_i_n)
                        # b_n = Interp_Mean(b_f_i, self.t, t_periods[n], t_periods[n+1])
                        # c_n = Interp_Mean(c_f_i, self.t, t_periods[n], t_periods[n+1])
                        self.evolutions[i][n,b_idx] = b_n
                        self.evolutions[i][n,c_idx] = c_n
                    # self.evolutions_f[i].append(A_f_i)
                    # self.evolutions_f[i].append(P_f_i)
            if lowcut != None:
                Fs_slow = 1/(t_periods[1] - t_periods[0])
                for m in self.order:
                    self.evolutions[i][:,m] = lowpass_filter(self.evolutions[i][:,m], lowcut[i], Fs)
            for m in self.order:
                if m == 0:
                    a0_ddot = np.gradient(np.gradient(self.evolutions[i][:,m], t_periods[:-1]))
                    self.dot[i][:,m] = a0_ddot + self.frequencys[i]**2*self.evolutions[i][:,m]
                else:
                    b_idx = self.harmonic_order[m][0]
                    c_idx = self.harmonic_order[m][1]
                    bm_dot = np.gradient(self.evolutions[i][:,b_idx],t_periods[:-1])
                    bm_ddot = np.gradient(bm_dot, t_periods[:-1])
                    cm_dot = np.gradient(self.evolutions[i][:,c_idx],t_periods[:-1])
                    cm_ddot = np.gradient(cm_dot, t_periods[:-1])
                    self.dot[i][:,b_idx] = bm_ddot + 2*self.frequencys[i]*cm_dot-((self.frequencys[i]*m)**2-self.frequencys[i]**2)*self.evolutions[i][:,b_idx] #cos--b*cos+c*sin
                    self.dot[i][:,c_idx] = cm_ddot - 2*self.frequencys[i]*bm_dot-((self.frequencys[i]*m)**2-self.frequencys[i]**2)*self.evolutions[i][:,c_idx] #sin--b*cos+c*sin
                    # self.dot[i][:,b_idx] = 2*Am_dot*(self.frequencys[i]*m+Pm_dot) + self.evolutions[i][:,b_idx]*Pm_ddot #cos--A*sin(t+phi)
                    # self.dot[i][:,c_idx] = Am_ddot-self.evolutions[i][:,b_idx]*((m*self.frequencys[i]*m+Pm_dot)**2-self.frequencys**2) #sin
    
    
    def Library_rebuild(self,F):
        from scipy.interpolate import interp1d
        self.Phi = []
        dt = self.t[1]-self.t[0]
        periods_discrete_supersample = 10000

        F_curve = interp1d(self.t, F, kind = 'linear')
        for i in range(self.dims):
            omega_i = self.frequencys[i]
            dt = self.t[1]-self.t[0]
            T = 2*np.pi/omega_i
            num_period = int((self.t[-1] - self.t[0]) // T)
            t_periods = np.linspace(T,T*num_period, num_period+1)
            self.Phi.append(np.zeros([num_period, self.harmonic_number, len(self.library)+1]))
            for n in range(num_period):
                states = []
                t_period = np.linspace(t_periods[n], t_periods[n+1], periods_discrete_supersample)
                
                dt_period = t_period[1] - t_period[0]
                F_period = F_curve(t_period)
                for j in range(self.dims):
                    states.append(self.data_f[j](t_period))
                    states.append(self.velocity_f[j](t_period))
                for l in range(len(self.library)):
                    library_lin = self.library[l](states)
                    for m in self.order:
                        if m == 0:
                            Phi_m_l = library_lin
                            self.Phi[i][n,m,l] = -np.trapz(Phi_m_l, t_period)/(2*np.pi)*self.frequencys[i]
                        else:
                            b_idx = self.harmonic_order[m][0]
                            c_idx = self.harmonic_order[m][1]
                            # P_i = Interp_Mean(self.evolutions_f[i][c_idx], self.t, t_period[0], t_period[-1])
                            Phi_b_l = library_lin*np.cos(self.frequencys[i]*m*t_period)
                            Phi_c_l = library_lin*np.sin(self.frequencys[i]*m*t_period)
                            self.Phi[i][n,b_idx,l] = -np.trapz(Phi_b_l, t_period)/(2*np.pi)*self.frequencys[i]*2
                            self.Phi[i][n,c_idx,l] = -np.trapz(Phi_c_l, t_period)/(2*np.pi)*self.frequencys[i]*2
                for m in self.order:
                    if m == 0:
                        Phi_m_l = F_period
                        self.Phi[i][n,m,-1] = -np.trapz(Phi_m_l, t_period)/(2*np.pi)*self.frequencys[i]
                    else:
                        b_idx = self.harmonic_order[m][0]
                        c_idx = self.harmonic_order[m][1]
                        # P_i = Interp_Mean(self.evolutions_f[i][c_idx], self.t, t_period[0], t_period[-1])
                        Phi_b_l = F_period*np.cos(self.frequencys[i]*m*t_period)
                        Phi_c_l = F_period*np.sin(self.frequencys[i]*m*t_period)
                        self.Phi[i][n,b_idx,-1] = -np.trapz(Phi_b_l, t_period)/(2*np.pi)*self.frequencys[i]*2
                        self.Phi[i][n,c_idx,-1] = -np.trapz(Phi_c_l, t_period)/(2*np.pi)*self.frequencys[i]*2
                
     
                        
    def Library_reschudle(self):
        from scipy.stats import pearsonr
        # remove the non-active basis functions which is almost 0 in evolutionary space
        # detect the linear-dependent basis functions and marked
        self.all_zeros = []
        self.to_drops = []
        self.correlation_groups = []
        for i in range(self.dims):
            all_zero = []
            to_drop = []
            correlation_group = []
            data_norm = np.linalg.norm(self.data[:,i])
            
            for j in range(np.shape(self.Phi[i])[2]):
                
                if np.linalg.norm(self.Phi[i][:,0,j])/len(self.Phi[i][:,0,j])< data_norm*self.zerobench and np.linalg.norm(self.Phi[i][:,1,j])/len(self.Phi[i][:,1,j])< data_norm*self.zerobench:
                    all_zero.append(j)
            for j in range(np.shape(self.Phi[i])[2]):
                if j in to_drop:
                    continue
                group = [j]
                for n in range(j+1, np.shape(self.Phi[i])[2]):
                    eff = 0
                    if n in to_drop:
                        continue
                        
                    for m in range(self.harmonic_number):
                        
                        corr_coeff, p_value = pearsonr(self.Phi[i][:,m,j], self.Phi[i][:,m,n])
                        if np.abs(corr_coeff) > self.corrbench:
                            eff+=1
                    if eff==self.harmonic_number:
                        to_drop.append(n)
                        group.append(n)
                        break

                if len(group)>1:
                    correlation_group.append(group)
            self.all_zeros.append(all_zero)
            self.to_drops.append(to_drop)
            self.correlation_groups.append(correlation_group)
            
            
        

    def optimize(self,orders,startcycle=0, endcycle = 1, sparse_threshold = 1e-3,stop_tolerance = 1e-4,step_tolerance = 1e-5,sparse_max = 6,smooth_window = 1,fixed = None):

        for i in range(self.dims):
            y = self.dot[i][startcycle:-endcycle,:]
            X = self.Phi[i][startcycle:-endcycle,:,:]
            Xi_i = cyxpy_solver(y, X,orders[i], lam = sparse_threshold ,stop_tolerance = stop_tolerance, step_tolerance = step_tolerance, sparse_max = sparse_max,smooth_window = 1,w_A2b = 1, fixed = None)
            # Xi_i = OMP(self.dot[i][startcycle:-endcycle,:],X,scale = self.library_scale[effidx], sparse_threshold = sparse_threshold ,stop_tolerance = stop_tolerance, step_tolerance = step_tolerance, sparse_max = sparse_max,smooth_window =smooth_window, fixed = fixed)
            self.Xi[i] = Xi_i
            # for j in range(len(effidx)):
            #     self.Xi[i][j] = Xi_i[j]         

    