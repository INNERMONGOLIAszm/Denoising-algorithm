# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 21:43:47 2021

@author: Zemin Shi
"""
import os
import matplotlib.pyplot as plt
import scipy.io as scio
import mne
import numpy as np
from mne.time_frequency import tfr_morlet, psd_multitaper, psd_welch
from mne.preprocessing import (ICA, create_eog_epochs, create_ecg_epochs,corrmap)
from mne.viz import plot_alignment
from scipy import optimize
from mne import transforms
from mne.utils import logger
import math
import copy

#Read raw data
datafile = 'E://IMAGE_processing//audi_evoked.fif'
raw33=raw = mne.io.read_raw_fif(datafile)

#Remove power frequency signals and harmonic noise
meg_picks = mne.pick_types(raw33.info, meg=True, eeg=False)  
freqs = (50,64,69)
raw2 = raw33.copy().notch_filter(freqs=freqs, picks=meg_picks)
#0.1 to 54 band pass filter
raw1 = raw2.copy()
raw0 = raw1.load_data().filter(l_freq=0.1, h_freq=54)

#Create a single-layer ball model
sphere = mne.make_sphere_model(r0=(0.00,0.00,0.00), head_radius=0.0000001)
#Create an internal point source
vol_src2_pos_inner_point={}
vol_src2_pos_inner_point['nn']=np.array([0,0,1])
vol_src2_pos_inner_point['rr']=np.array([0.0,  0.02 ,  0.075])
vol_src2_pos_inner_point['nn']=copy.copy(vol_src2_pos_inner_point['nn'][np.newaxis,:])
vol_src2_pos_inner_point['rr']=copy.copy(vol_src2_pos_inner_point['rr'][np.newaxis,:])
vol_src_new_inner_point = mne.setup_volume_source_space(mri=None,sphere=None,pos=vol_src2_pos_inner_point)
fwd_new_inner_point = mne.make_forward_solution(raw0.info, trans=None, src=vol_src_new_inner_point,bem=sphere,
                                meg=True, eeg=False, mindist=5.0, n_jobs=2)
#Create an external point source
vol_src2_pos_out_point={}
vol_src2_pos_out_point['nn']=np.array([0,0,1])
vol_src2_pos_out_point['rr']=np.array([ 0.005, -0.035, -3.11 ])
vol_src2_pos_out_point['nn']=copy.copy(vol_src2_pos_out_point['nn'][np.newaxis,:])
vol_src2_pos_out_point['rr']=copy.copy(vol_src2_pos_out_point['rr'][np.newaxis,:])
vol_src_new_out_point = mne.setup_volume_source_space(subject='sample', bem=sphere,mri=None,
                                        sphere=None,pos=vol_src2_pos_out_point)
fwd_new_out_point = mne.make_forward_solution(raw0.info, trans=None, src=vol_src_new_out_point,bem=sphere,
                                meg=True, eeg=False, mindist=5.0, n_jobs=2)

#Impulse function
def impseq(times,pulse,interval,amplitude):
    a = np.zeros(times)
    n=int(times/(pulse+interval))
    for i in range(n):
        a[(interval*(i+1)+pulse*i):(interval*(i+1)+pulse*(i+1))] = 1
        if (interval*(i+1)+pulse*(i+1))>times:
            break
    a=amplitude*a
    return a

#Dipole sends sin signal
from mne.time_frequency import fit_iir_model_raw
from mne.viz import plot_sparse_source_estimates
from mne.simulation import simulate_sparse_stc, simulate_evoked
times = np.arange(raw0['data'][1].shape[0], dtype=np.float) / raw0.info['sfreq'] - 0.1
rng = np.random.RandomState(42)
def data_fun(times):
    """Function to generate random source time courses"""
    return (20e-4 * impseq(len(times),10,30,6) + 25e-13*0.5 * rng.randn(1))

#Dipole sends pulse signal
def data_fun_inner(times):
    """Function to generate random source time courses"""
    return (14e-7 * np.sin(30. * times) + 50e-14*0.5 * rng.randn(1))


from mne.simulation import (simulate_sparse_stc, simulate_raw,
                            add_noise, add_ecg, add_eog)
#Create external simulation signal
stc_out_point = simulate_sparse_stc(vol_src_new_out_point, n_dipoles=1, times=times,
                          random_state=1, data_fun=data_fun)
raw_sim_out_point = simulate_raw(raw0.info,
                       stc_out_point,
                       forward=fwd_new_out_point,
                       verbose=True)
#Create inner simulation signal
stc_inner_point = simulate_sparse_stc(vol_src_new_inner_point, n_dipoles=1, times=times,
                          random_state=1, data_fun=data_fun_inner)
raw_sim_inner_point = simulate_raw(raw0.info,
                       stc_inner_point,
                       forward=fwd_new_inner_point,
                       verbose=True)
raw_sim_inner_point.plot(n_channels=5,scalings=0.1*10e-11,duration=4)

#Create a simulated signal with spatial noise
raw_stimu_noise_data=raw_sim_out_point['data'][0]+raw_sim_inner_point['data'][0]
raw_stimu_noise_time=raw_sim_out_point['data'][1]
raw_stimu_noise = mne.io.RawArray(raw_stimu_noise_data, raw0.info)

#The mean square error between the data after SSS denoising and
#the true value is taken as the objective function. Minimum objective
#function. Inverse solution parameter
from scipy.optimize import minimize
import numpy as np
def find_SSS_parameter(A,B,lint,lout):
    def fun():
        v=lambda x: rawSSS_variance(A,B,x[0],x[1],x[2],lint,lout)
        return v
    def con(args):
        # 约束条件 分为eq 和ineq
        #eq表示 函数结果等于0 ； ineq 表示 表达式大于等于0  
        x1min, x1max, x2min, x2max,x3min,x3max = args
        cons = ({'type': 'ineq', 'fun': lambda x: x[0] - x1min},\
                  {'type': 'ineq', 'fun': lambda x: -x[0] + x1max},\
                 {'type': 'ineq', 'fun': lambda x: x[1] - x2min},\
                    {'type': 'ineq', 'fun': lambda x: -x[1] + x2max},\
                {'type': 'ineq', 'fun': lambda x: x[2] - x3min},\
                 {'type': 'ineq', 'fun': lambda x: -x[2] + x3max})
        return cons

    if __name__ == "__main__":
        #设置参数范围/约束条件
        args1 = (-0.08,0.08,-0.08,0.08,-0.1,0.1)  #x1min, x1max, x2min, x2max
        cons = con(args1)
        #设置初始猜测值  
        x0 = np.asarray((0.0,0.0,0.0))
        res = minimize(fun(), x0, method='SLSQP',constraints=cons)
        print(res.fun)
        print(res.success)
        print(res.x)
    return res.x

#Calculate the SSS of different parameters,
#the mean square error of A and B after denoising
def rawSSS_variance(A, B, x, y, z, a, b):
    try:
        raw_sss_104chan = mne.preprocessing.maxwell_filter(A, int_order=a, ext_order=b,
        verbose=True, coord_frame="meg", origin=[x, y, z])
        Mean_square_var = rawXY_variance(raw_sss_104chan, B)
    except:
        Mean_square_var = 30
        pass
    return Mean_square_var * 10e8

#The mean square error of the first 2 seconds 
#of the two raw data (to judge the similarity)
def rawXY_variance(X,Y):
    X.crop(tmax=2).load_data() 
    Y.crop(tmax=2).load_data()
    total_variance=0
    for i in range(Y.info['nchan']):
        variance=np.sum(np.square(X['data'][0][i]-Y['data'][0][i]))
        total_variance+=variance
    total_variance=total_variance/(Y.info['nchan'])
    Mean_square_var=np.sqrt(total_variance)
    return Mean_square_var

#Automatically find the best parameters
# for i in range(5):
#     for j in range(5):
#         find_SSS_parameter(raw_stimu_noise,raw_sim_inner_point,i,j)
#         print("##################END：lin is %s#######lout is %s"%(i,j))

#The best parameters were found to be: int_order=3, ext_order=1,
#origin=[0.02301067 ,0.01993269, 0.04286656]

#Beihang raw data, add impulse noise in space
raw_add_noise_data=raw0['data'][0]+raw_sim_out_point['data'][0]
raw_add_noise_time=raw_sim_out_point['data'][1]
raw_add_noise = mne.io.RawArray(raw_add_noise_data, raw0.info)

#SSS filtering after adding noise
raw_sss_rm = mne.preprocessing.maxwell_filter(raw_add_noise,int_order=3, ext_order=1,origin=[0.02301067 ,0.01993269, 0.04286656] ,st_duration=30,st_correlation=0.6,verbose=True,coord_frame="meg")
raw_sss_rm.plot(n_channels=5,scalings=0.1*10e-11,duration=4)
raw_sss_rm.pick(['mag']).plot(duration=4,scalings=0.1*10e-11,butterfly=True)
#Excellent denoising effect

#SSS spatial filtering with optimal parameters on the original data
raw_sss = mne.preprocessing.maxwell_filter(raw0,int_order=3, ext_order=1,st_duration=30,st_correlation=0.8,
                                                  origin=[0.02301067 ,0.01993269, 0.04286656],verbose=True,coord_frame="meg")

#Comparison of the four butterfly charts
raw_add_noise.pick(['mag']).plot(duration=4,scalings=0.1*10e-11,butterfly=True)
raw0.pick(['mag']).plot(duration=4,scalings=0.1*10e-11,butterfly=True)
raw_sss.pick(['mag']).plot(duration=4,scalings=0.1*10e-11,butterfly=True)
raw_sss_rm.pick(['mag']).plot(duration=4,scalings=0.1*10e-11,butterfly=True)


#Stimulus time sequence
event_dict={'auditory/left':1}
events = mne.find_events(raw0,stim_channel='STI 014',uint_cast=True,initial_event = True)
events_new=copy.copy(events[:,:])
event_shape=np.zeros(np.array(events_new).shape).astype(int)
event_shape[:,0]=200
events_new=copy.copy(events_new+event_shape)

#Form epochs slice
epochs_raw0 = mne.Epochs(raw0, events=events_new,tmin=-0.1,event_id=event_dict,proj=True,
                    tmax=0.4,  baseline=(None,0),detrend=1, 
                    verbose=True)
epochs_rawsss = mne.Epochs(raw_sss, events=events_new,tmin=-0.1,event_id=event_dict,proj=True,
                    tmax=0.4,  baseline=(None,0),detrend=1, 
                    verbose=True)

#Draw evoked graph
evokeds_raw0 = epochs_raw0.average().pick('mag')
evokeds_rawsss = epochs_rawsss.average().pick('mag')

#Drawing display
evokeds_raw0.plot_joint(picks='mag')
evokeds_rawsss.plot_joint(picks='mag')


subject = 'raw_scaled'
data_path = 'E:\IMAGE_processing\data\MNE-sample-data-processed'
subjects_dir = data_path + '/freesurferICBM152'
bem_dir=data_path + '/bem_ICBM_152_scaled.fif'
fwd_dir_surface=data_path + '/fwd_ICBM_152_scaled_surface.fif'
fwd_ICBM_read_surface = mne.read_forward_solution(fwd_dir_surface)


#Mapping cortical sources
from mne.minimum_norm import make_inverse_operator, apply_inverse
from mne.viz import plot_sparse_source_estimates
noise_cov = mne.compute_covariance(epochs_rawsss,tmax=0.0,method='auto',rank=None)
snr = 9
lambda2 = 1.0 / snr ** 2
inverse_operator = make_inverse_operator(evokeds_rawsss.info, fwd_ICBM_read_surface,noise_cov , loose=1.0 , depth=0.1)
surfer_kwargs = dict(
    hemi='rh', subjects_dir=subjects_dir,
    clim=dict(kind='value', lims=[2, 3, 5]), 
    views='lateral',time_unit='s', size=(800, 800), smoothing_steps=9)
surfer_kwargs['clim'].update(kind='percent', lims=[82.68, 88.88, 92.98])
method='eLORETA'
#method in enumerate(['dSPM', 'sLORETA', 'eLORETA'])
stc = apply_inverse(evokeds_rawsss, inverse_operator, lambda2,
                    method=method, pick_ori=None,
                    verbose=True)
brain = stc.plot(figure=1, **surfer_kwargs)
brain.add_text(0.1, 0.9, method, 'title', font_size=20)

#Use rap_music for source location
from mne.beamformer import rap_music
from mne.viz import plot_dipole_locations, plot_dipole_amplitudes
dipoles, residual = rap_music(evokeds_rawsss.copy().crop(-0.1,0.4), fwd_ICBM_read_surface,noise_cov, n_dipoles=1,return_residual=True, verbose=True)
trans_rap_music = fwd_ICBM_read_surface['mri_head_t']
plot_dipole_locations(dipoles, trans_rap_music,'raw_scaled', subjects_dir=subjects_dir)

















