from mountainlab_pytools import mdaio
import numpy as np
import multiprocessing
import time
import os

processor_name = 'ephys.mask_out_artifacts'
processor_version = '0.1.0'


class SharedChunkInfo():
    def __init__(self, num_chunks):
        self.timer_timestamp = multiprocessing.Value('d', time.time(), lock=False)
        self.last_appended_chunk = multiprocessing.Value('l', -1, lock=False)
        self.num_chunks = num_chunks
        self.num_completed_chunks = multiprocessing.Value('l', 0, lock=False)
        self.lock = multiprocessing.Lock()

    def reportChunkCompleted(self, num):
        with self.lock:
            self.num_completed_chunks.value += 1

    def reportChunkAppended(self, num):
        with self.lock:
            self.last_appended_chunk.value = num

    def lastAppendedChunk(self):
        with self.lock:
            return self.last_appended_chunk.value

    def resetTimer(self):
        with self.lock:
            self.timer_timestamp.value = time.time()

    def elapsedTime(self):
        with self.lock:
            return time.time() - self.timer_timestamp.value

    def printStatus(self):
        with self.lock:
            print('Processed {} of {} chunks...'.format(self.num_completed_chunks.value, self.num_chunks))


def mask_out_artifacts(*, timeseries, timeseries_out, threshold=6, chunk_size=2000, num_write_chunks=150,
                         num_processes=os.cpu_count()):
    """
    Masks out artifacts. Each chunk will be analyzed, and if the square root of the
    RSS of the chunk is above threshold, all the samples in this chunk (and neighboring chunks)
    will be set to zero.

    Parameters
    ----------
    timeseries : INPUT
        MxN raw timeseries array (M = #channels, N = #timepoints)

    timeseries_out : OUTPUT
        masked output (MxN array)

    threshold : int
        Number of standard deviations away from the mean to consider as artifacts (default of 6).
    chunk_size : int
        This chunk size will be the number of samples that will be set to zero if the square root RSS of this chunk is above threshold.
    num_write_chunks : int
        How many chunks will be simultaneously written to the timeseries_out path (default of 150).
    """

    if threshold == 0 or chunk_size == 0 or num_write_chunks == 0:
        print("Problem with input parameters. Either threshold, num_write_chunks, or chunk_size is zero.\n")
        return False

    write_chunk_size = chunk_size * num_write_chunks

    opts = {
        "timeseries": timeseries,
        "timeseries_out": timeseries_out,
        "chunk_size": chunk_size,
        "num_processes": num_processes,
        "num_write_chunks": num_write_chunks,
        "write_chunk_size": write_chunk_size,
    }

    global g_opts
    g_opts = opts

    X = mdaio.DiskReadMda(timeseries)

    M = X.N1()  # Number of channels
    N = X.N2()  # Number of timepoints

    # compute norms of chunks
    num_chunks = int(np.ceil(N / chunk_size))
    num_write = int(np.ceil(N / write_chunk_size))

    norms = np.zeros((M, num_chunks))  # num channels x num_chunks

    for i in np.arange(num_chunks):
        t1 = int(i * chunk_size)  # first timepoint of the chunk
        t2 = int(np.minimum(N, (t1 + chunk_size)))  # last timepoint of chunk (+1)

        chunk = X.readChunk(i1=0, N1=X.N1(), i2=t1, N2=t2 - t1).astype(np.float32)  # Read the chunk

        norms[:, i] = np.sqrt(np.sum(chunk ** 2, axis=1))  # num_channels x num_chunks

    # determine which chunks to use
    use_it = np.ones(num_chunks)  # initialize use_it array

    for m in np.arange(M):
        vals = norms[m, :]

        sigma0 = np.std(vals)
        mean0 = np.mean(vals)

        artifact_indices = np.where(vals > mean0 + sigma0 * threshold)[0]

        # check if the first chunk is above threshold, ensure that we don't use negative indices later
        negIndBool = np.where(artifact_indices > 0)[0]

        # check if the last chunk is above threshold to avoid a IndexError
        maxIndBool = np.where(artifact_indices < num_chunks - 1)[0]

        use_it[artifact_indices] = 0
        use_it[artifact_indices[negIndBool] - 1] = 0  # don't use the neighbor chunks either
        use_it[artifact_indices[maxIndBool] + 1] = 0  # don't use the neighbor chunks either

        print("For channel %d: mean=%.2f, stdev=%.2f, chunk size = %d\n" % (m, mean0, sigma0, chunk_size))

    global g_shared_data
    g_shared_data = SharedChunkInfo(num_write)

    mdaio.writemda32(np.zeros([M, 0]), timeseries_out)  # create initial file w/ empty array so we can append to it

    pool = multiprocessing.Pool(processes=num_processes)
    # pool.starmap(mask_chunk,[(num,use_it[num]) for num in range(0,num_chunks)],chunksize=1)
    pool.starmap(mask_chunk, [(num, use_it[num * num_write_chunks:(num + 1) * num_write_chunks]
                               ) for num in range(0, num_write)], chunksize=1)

    num_timepoints_used = sum(use_it)
    num_timepoints_not_used = sum(use_it == 0)
    print("Using %.2f%% of all timepoints.\n" % (
    num_timepoints_used * 100.0 / (num_timepoints_used + num_timepoints_not_used)))
    return True


def mask_chunk(num, use_it):
    opts = g_opts

    in_fname = opts['timeseries']  # The entire (large) input file
    out_fname = opts['timeseries_out']  # The entire (large) output file
    chunk_size = opts['chunk_size']
    num_write_chunks = opts['num_write_chunks']
    write_chunk_size = opts['write_chunk_size']

    X = mdaio.DiskReadMda(in_fname)

    # t1=int(num*chunk_size) # first timepoint of the chunk
    # t2=int(np.minimum(X.N2(),(t1+chunk_size))) # last timepoint of chunk (+1)

    t1 = int(num * write_chunk_size)  # first timepoint of the chunk
    t2 = int(np.minimum(X.N2(), (t1 + write_chunk_size)))  # last timepoint of chunk (+1)

    chunk = X.readChunk(i1=0, N1=X.N1(), i2=t1, N2=t2 - t1).astype(np.float32)  # Read the chunk

    if sum(use_it) != len(use_it):
        idmax = t2 - t1
        idx = get_masked_indices(use_it, write_chunk_size, chunk_size, num_write_chunks)
        if idx[-1]>=idmax:
            idx = idx[idx < idmax]
        chunk[:, idx] = 0

    ###########################################################################################
    # Now we wait until we are ready to append to the output file
    # Note that we need to append in order, thus the shared_data object
    ###########################################################################################
    g_shared_data.reportChunkCompleted(num)  # Report that we have completed this chunk
    while True:
        if num == g_shared_data.lastAppendedChunk() + 1:
            break
        time.sleep(0.005)  # so we don't saturate the CPU unnecessarily

    # Append the filtered chunk (excluding the padding) to the output file
    mdaio.appendmda(chunk, out_fname)

    # Report that we have appended so the next chunk can proceed
    g_shared_data.reportChunkAppended(num)

    # Print status if it has been long enough
    if g_shared_data.elapsedTime() > 4:
        g_shared_data.printStatus()
        g_shared_data.resetTimer()


def get_masked_indices(use_it, write_chunk_size, chunk_size, num_write_chunks):
    indices = np.arange(write_chunk_size).reshape((num_write_chunks, chunk_size))
    return indices[np.where(use_it==0)[0], :].flatten() # fix by jfm 9/5/18

def test_mask_out_artifacts():
    
    # Create noisy array
    samplerate = int(48e3)
    duration = 30 # seconds
    n_samples = samplerate*duration
    noise_amplitude = 5
    noise = noise_amplitude*np.random.normal(0,1,n_samples)
    standard_dev = np.std(noise)
    
     # add three artefacts
    n_artifacts = 3
    artifacts = np.zeros_like(noise)
    artifact_duration = int(0.2*samplerate) # samples
    artifact_signal = np.zeros((n_artifacts, artifact_duration))

    for i in np.arange(n_artifacts):                   
        artifact_signal[i, :] = noise_amplitude*np.random.normal(0,6,artifact_duration)

    artifact_indices = np.tile(np.arange(artifact_duration), (3,1))

    artifact_shift = np.array([int(n_samples*0.10), int(n_samples*0.20), int(n_samples*0.70)])

    artifact_indices += artifact_shift.reshape((-1,1))

    for i, indices in enumerate(artifact_indices):
        artifacts[indices] = artifact_signal[i,:]

    signal = noise + artifacts

    timeseries = 'test_mask.mda'
    timeseries_out = 'masked.mda' 
    
    # write as mda
    mdaio.writemda32(signal.reshape((1,-1)), timeseries)
    
    # run the mask artefacts
    mask_out_artifacts(timeseries=timeseries, timeseries_out=timeseries_out, threshold=6, chunk_size=2000, 
                       num_write_chunks=150)
    
    # check that they are gone 
    read_data = mdaio.readmda(timeseries).reshape((-1,1))
    masked_data = mdaio.readmda(timeseries_out).reshape((-1,1))

    indices_masked = sum(masked_data[artifact_indices,0].flatten() == 0)
    total_indices_to_mask = len(artifact_indices.flatten())
    masked = indices_masked == total_indices_to_mask
    
    os.remove(timeseries)
    os.remove(timeseries_out)
    
    if masked:
        print('Artifacts 100% masked')
        return True
    else:
        print('Artifacts %.2f%% masked' % (100*(indices_masked/total_indices_to_mask)))
        return False

mask_out_artifacts.name = processor_name
mask_out_artifacts.version = processor_version
mask_out_artifacts.author = "Geoffrey Barrett"
