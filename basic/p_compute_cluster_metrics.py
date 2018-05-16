import numpy as np

from mltools import mdaio
from p_compute_templates import compute_templates_helper
from p_compute_cross_correlograms import compute_cross_correlograms_helper
import json

processor_name='ephys.compute_cluster_metrics'
processor_version='0.1'
def compute_cluster_metrics(*,timeseries='',firings,metrics_out,clip_size=100,samplerate=0,
        compute_auto_correl=False):
    """
    Compute cluster metrics for a spike sorting output

    Parameters
    ----------
    firings : INPUT
        Path of firings mda file (RxL) where R>=3 and L is the number of events. Second row are timestamps, third row are integer cluster labels.
    timeseries : INPUT
        Optional path of timeseries mda file (MxN) which could be raw or preprocessed   
        
    metrics_out : OUTPUT
        Path of output json file containing the metrics.
        
    clip_size : int
        (Optional) clip size, aka snippet size (used when computing the templates, or average waveforms)
    samplerate : float
        Optional sample rate in Hz
    """    
    print('Reading firings...')
    F=mdaio.readmda(firings)

    print('Initializing...')
    R=F.shape[0]
    L=F.shape[1]
    assert(R>=3)
    times=F[1,:]
    labels=F[2,:].astype(np.int)
    K=np.max(labels)
    N=0
    if timeseries:
        X=mdaio.DiskReadMda(timeseries)
        N=X.N2()

    if (samplerate>0) and (N>0):
        duration_sec=N/samplerate
    else:
        duration_sec=0

    clusters=[]
    for k in range(1,K+1):
        inds_k=np.where(labels==k)[0]
        metrics_k={
            "num_events":len(inds_k)
        }
        if duration_sec:
            metrics_k['firing_rate']=len(inds_k)/duration_sec
        cluster={
            "label":k,
            "metrics":metrics_k
        }
        clusters.append(cluster)

    if timeseries:
        print('Computing templates...')
        templates=compute_templates_helper(timeseries=timeseries,firings=firings,clip_size=clip_size)
        for k in range(1,K+1):
            template_k=templates[:,:,k-1]
            # subtract mean on each channel (todo: vectorize this operation)
            # Alex: like this?
            # template_k[m,:] -= np.mean(template_k[m,:])
            for m in range(templates.shape[0]):
                template_k[m,:]=template_k[m,:]-np.mean(template_k[m,:])
            peak_amplitude=np.max(np.abs(template_k))
            clusters[k-1]['metrics']['peak_amplitude']=peak_amplitude
        ## todo: subtract template means, compute peak amplitudes

    if compute_auto_correl:
        print('Computing Auto-Correlegrams')
        auto_cors = compute_cross_correlograms_helper(firings,mode='autocorrelograms',samplerate=samplerate,max_dt_msec=50,bin_size_msec=2)
        mid_ind  = mat_dt_msec/bins_size_msec/2
        mid_inds = np.arange(mid_ind-2,mid_ind+2)
        for k0,auto_cor in enumerate(auto_cors):
            peak = np.percentile(auto_cor, 75)*1.0
            mid  = np.mean(auto_cor[mid_inds])
            clusters[k]['metrics']['refractory_ratio'] = 1-(mid/peak)

    ret={
        "clusters":clusters
    }

    print('Writing output...')
    str=json.dumps(ret,indent=4)
    with open(metrics_out, 'w') as out:
        out.write(str)
    print('Done.')
    return True

compute_cluster_metrics.name=processor_name
compute_cluster_metrics.version=processor_version
