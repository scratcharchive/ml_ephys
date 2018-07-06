import numpy as np
import h5py
import deepdish as dd

from mountainlab_pytools import mdaio
from timeserieschunkreader import TimeseriesChunkReader

processor_name='ephys.compute_cross_correlograms'
processor_version='0.11'
def compute_cross_correlograms(*,firings,correlograms_out,samplerate,mode='autocorrelograms',max_dt_msec=50,bin_size_msec=2):
    """
    Compute cross-correlograms for a firings file

    Parameters
    ----------
    firings : INPUT
        Path of firings mda file (RxL) where R>=3 and L is the number of events. Second row are timestamps, third row are integer labels.    
        
    correlograms_out : OUTPUT
        Path of output hdf5 file
    
    mode : string
        Choices: autocorrelograms, (more to come)    
    samplerate : float
        Sample rate in Hz
    max_dt_msec : float
        Max dt for the histograms (msec)
    bin_size_msec : float
        Bin size for the histograms (msec)
    """    
    X=compute_cross_correlograms_helper(firings=firings,mode=mode,samplerate=samplerate,max_dt_msec=max_dt_msec,bin_size_msec=bin_size_msec)
    dd.io.save(correlograms_out,X)
    return True
    
# Same as compute_cross_correlograms, except return the result in memory
def compute_cross_correlograms_helper(*,firings,mode='autocorrelograms',samplerate=30000,max_dt_msec=50,bin_size_msec=2):
    if type(firings)==str:
        F=mdaio.readmda(firings)
    else:
        F=firings
    R,L=np.shape(F)
    assert(R>=3)
    assert(mode=='autocorrelograms')
    max_dt_tp=max_dt_msec/1000*samplerate
    bin_size_tp=bin_size_msec/1000*samplerate
    times=F[1,:]
    labels=F[2,:].astype(int)
    K=np.max(labels)
    correlograms=[]
    for k in range(1,K+1):
        inds_k=np.where(labels==k)[0]
        times_k=times[inds_k]
        bin_counts,bin_edges=compute_autocorrelogram(times_k,max_dt_tp=max_dt_tp,bin_size_tp=bin_size_tp)
        correlograms.append({
            "k":k,
            "bin_edges":bin_edges/samplerate*1000,
            "bin_counts":bin_counts
        })
    return {
        "correlograms":correlograms
    }

def compute_autocorrelogram(times,*,max_dt_tp,bin_size_tp):
    num_bins_left=int(max_dt_tp/bin_size_tp)
    L=len(times)
    times2=np.sort(times)
    step=1
    candidate_inds=np.arange(L)
    vals_list=[]
    while True:
        candidate_inds=candidate_inds[candidate_inds+step<L]
        candidate_inds=candidate_inds[times2[candidate_inds+step]-times2[candidate_inds]<=max_dt_tp]
        if len(candidate_inds)>0:
            vals=times2[candidate_inds+step]-times2[candidate_inds]
            vals_list.append(vals)
            vals_list.append(-vals)
        else:
            break
        step+=1
    if len(vals_list)>0:
        all_vals=np.concatenate(vals_list)
    else:
        all_vals=np.array([]);
    aa=np.arange(-num_bins_left,num_bins_left+1)*bin_size_tp
    all_vals=np.sign(all_vals)*(np.abs(all_vals)-bin_size_tp*0.00001) # a trick to make the histogram symmetric
    bin_counts,bin_edges=np.histogram(all_vals,bins=aa)
    return (bin_counts,bin_edges)
    
compute_cross_correlograms.name=processor_name
compute_cross_correlograms.version=processor_version
