import numpy as np

from mltools import mdaio
import json

processor_name='ephys.compare_ground_truth'
processor_version='0.1'

def create_occupancy_array(times,labels,segment_size,num_segments,K,*,spread,multiplicity):
    ret=np.zeros((K,num_segments),dtype=np.int)
    counts=np.zeros(K,dtype=np.int64)
    for k in range(1,K+1):
        inds_k=np.where(labels==k)[0]
        times_k=times[inds_k]
        labels_k=labels[inds_k]
        segment_inds=np.floor(times_k/segment_size).astype(np.int64)
        tmp=np.bincount(segment_inds)
        if not multiplicity:
            tmp[np.where(ret>=1)[0]]=1
        ret[k-1,0:len(tmp)]=tmp
        if spread:
            ret[k-1,np.maximum(0,segment_inds-1)]=1
            ret[k-1,np.minimum(num_segments-1,segment_inds+1)]=1
        counts[k-1]=len(inds_k)
    return (ret,counts)

def compare_ground_truth(*,firings_true,firings,json_out,max_dt=20):
    """
    compare a sorting (firings) with ground truth (firings_true)

    Parameters
    ----------
    firings_true : INPUT
        Path of true firings file (RxL) R>=3, L=#evts
    firings : INPUT
        Path of sorted firings file (RxL) R>=3, L=#evts
    json_out : OUTPUT
        Path of the output file containing the results of the comparison
        
    max_dt : int
        Tolerance for matching events (in timepoints)
        
    """

    print('Reading arrays...')
    F=mdaio.readmda(firings)
    Ft=mdaio.readmda(firings_true)
    print('Initializing data...')
    L=F.shape[1]
    Lt=Ft.shape[1]
    times=F[1,:]
    times_true=Ft[1,:]
    labels=F[2,:].astype(np.int32)
    labels_true=Ft[2,:].astype(np.int32)
    F=0 # free memory?
    Ft=0 # free memory?
    N=np.maximum(np.max(times),np.max(times_true))
    K=np.max(labels)
    Kt=np.max(labels_true)

    # todo: subsample in first pass to get the best
    
    # First we split into segments
    print('Splitting into segments')
    segment_size=max_dt
    num_segments=int(np.ceil(N/segment_size))
    N=num_segments*segment_size
    segments=[]
    # occupancy: K x num_segments
    occupancy,counts=create_occupancy_array(times,labels,segment_size,num_segments,K,spread=False,multiplicity=True)
    # occupancy_true: Kt x num_segments
    occupancy_true,counts_true=create_occupancy_array(times_true,labels_true,segment_size,num_segments,Kt,spread=True,multiplicity=False)
    
    # Note: we spread the occupancy_true but not the occupancy
    # Note: we count the occupancy with multiplicity but not occupancy_true
    
    print('Computing pairwise counts and accuracies...')
    pairwise_counts=occupancy_true @ occupancy.transpose() # Kt x K
    pairwise_accuracies=np.zeros((Kt,K))
    for k1 in range(1,Kt+1):
        for k2 in range(1,K):
            numer=pairwise_counts[k1-1,k2-1]
            denom=counts_true[k1-1]+counts[k2-1]-numer
            if denom>0:
                pairwise_accuracies[k1-1,k2-1]=numer/denom
    
    print('Preparing output...')
    ret={
        "true_units":{}
    }
    for k1 in range(1,Kt+1):
        k2_match=int(1+np.argmax(pairwise_accuracies[k1-1,:].ravel()))
        # todo: compute accuracy more precisely here
        num_matches=int(pairwise_counts[k1-1,k2_match-1])
        num_false_positives=int(counts[k2_match-1]-num_matches)
        num_false_negatives=int(counts_true[k1-1]-num_matches)
        unit={
            "best_match":k2_match,
            "accuracy":pairwise_accuracies[k1-1,k2_match-1],
            "num_matches":num_matches,
            "num_false_positives":num_false_positives,
            "num_false_negatives":num_false_negatives
        }
        ret['true_units'][k1]=unit
        
    print('Writing output...')
    str=json.dumps(ret,indent=4)
    with open(json_out, 'w') as out:
        out.write(str)
    print('Done.')
    
    return True

compare_ground_truth.name=processor_name
compare_ground_truth.version=processor_version