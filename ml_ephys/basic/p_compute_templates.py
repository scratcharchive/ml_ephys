import numpy as np

from mountainlab_pytools import mdaio
from timeserieschunkreader import TimeseriesChunkReader

processor_name='ephys.compute_templates'
processor_version='0.11'
def compute_templates(*,timeseries,firings,templates_out,clip_size=100):
    """
    Compute templates (average waveforms) for clusters defined by the labeled events in firings.

    Parameters
    ----------
    timeseries : INPUT
        Path of timeseries mda file (MxN) from which to draw the event clips (snippets) for computing the templates. M is number of channels, N is number of timepoints.
    firings : INPUT
        Path of firings mda file (RxL) where R>=3 and L is the number of events. Second row are timestamps, third row are integer labels.    
        
    templates_out : OUTPUT
        Path of output mda file (MxTxK). T=clip_size, K=maximum cluster label. Note that empty clusters will correspond to a template of all zeros. 
        
    clip_size : int
        (Optional) clip size, aka snippet size, number of timepoints in a single template
    """    
    templates=compute_templates_helper(timeseries=timeseries,firings=firings,clip_size=clip_size)
    return mdaio.writemda32(templates,templates_out)
    
# Same as compute_templates, except return the templates as an array in memory
def compute_templates_helper(*,timeseries,firings,clip_size=100):
    X=mdaio.DiskReadMda(timeseries)
    M,N = X.N1(),X.N2()
    F=mdaio.readmda(firings)
    L=F.shape[1]
    L=L
    T=clip_size
    Tmid = int(np.floor((T + 1) / 2) - 1);
    times=F[1,:].ravel()
    labels=F[2,:].ravel().astype(int)
    K=np.max(labels)

    sums=np.zeros((M,T,K),dtype='float64')
    counts=np.zeros(K)

    for k in range(1,K+1):
        inds_k=np.where(labels==k)[0]
        #TODO: subsample
        for ind_k in inds_k:
            t0=int(times[ind_k])
            if (clip_size<=t0) and (t0<N-clip_size):
                clip0=X.readChunk(i1=0,N1=M,i2=t0-Tmid,N2=T)
                sums[:,:,k-1]+=clip0
                counts[k-1]+=1
    templates=np.zeros((M,T,K))
    for k in range(K):
        templates[:,:,k]=sums[:,:,k]/counts[k]
    return templates
    
compute_templates.name=processor_name
compute_templates.version=processor_version
def test_compute_templates():
    M,N,K,T,L = 5,1000,6,50,100
    X=np.random.rand(M,N)
    mdaio.writemda32(X,'tmp.mda')
    F=np.zeros((3,L))
    F[1,:]=1+np.random.randint(N,size=(1,L))
    F[2,:]=1+np.random.randint(K,size=(1,L))
    mdaio.writemda64(F,'tmp2.mda')
    ret=compute_templates(timeseries='tmp.mda',firings='tmp2.mda',templates_out='tmp3.mda',clip_size=T)
    assert(ret)
    templates0=mdaio.readmda('tmp3.mda')
    assert(templates0.shape==(M,T,K))
    return True
compute_templates.test=test_compute_templates
