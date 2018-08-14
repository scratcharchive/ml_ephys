#!/usr/bin/env python3

from mountainlab_pytools.mlprocessors.registry import registry, register_processor

registry.namespace = "ephys_new"

from validation.p_compare_ground_truth import compare_ground_truth as original_compare_ground_truth
from mountainlab_pytools.mlprocessors.core import Input, Output, Processor, IntegerParameter, FloatParameter
from mountainlab_pytools.mlprocessors.validators import Validator, FileExtensionValidator

class MdaInput(Input):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.validators.insert(0, FileExtensionValidator(["mda"]))


class MdaShapeValidatorBase(Validator):
    def __init__(self, expected = None):
        self.expected = expected

    def validate(self, value):
        mda = mdaio._read_header(value)
        if not mda:
            raise ValidationError("Can't read mda header")
        if not self.validate_shape(value, mda.dims):
            if self.expected:
                raise ValidationError('Expected MDA shape is {}'.format(self.expected))
            else:
                raise ValidationError('MDA file does not have proper shape')

    def validate_shape(self, value, dims):
        return True



class MdaShapeValidator(Validator):
    def __init__(self, *, **kwargs):
        self.constraints = kwargs

    def validate(self, value):
        from mountainlab_pytools import mdaio
        mda = mdaio._read_header(value)
        if not mda:
            raise ValidationError("Can't read mda header")
#        dims = mda.dims
        if 'mindims' in self.constraints:
            mindims = self.constraints['mindims']
            if mda.num_dims < mindims:
                raise ValidationError("Number of dimensions has to be at least {}".format(mindims))
        if 'maxdims' in self.constraints:
            maxdims = self.constraints['maxdims']
            if mda.num_dims > maxdims:
                raise ValidationError("Number of dimensions has to be at most {}".format(maxdims))
        if 'dims' in self.constraints:
            dims = self.constraints('dims')
            if len(dims) == 1 and mda.num_dims != dims[0]:
                raise ValidationError("Number of dimensions has to be {}".format(dims[0]))
            elif len(dims) == 2:
                if dims[0] and mda.num_dims < dims[0]:
                    raise ValidationError("Number of dimensions has to be at least {}".format(dims[0]))
                if dims[1] and mda.num_dims > dims[1]:
                    raise ValidationError("Number of dimensions has to be at most {}".format(dims[1]))
            else:
                raise ValueError('Dimension constraint is not defined correctly.')


@register_processor(registry)
class compare_ground_truth(Processor):
    """
        compare a sorting (firings) with ground truth (firings_true)
    """
    VERSION=0.1

    firings_true = Input('Path of true firings file (RxL) R >=3, L=#evts')
    firings      = Input('Path of sorted firings file (RxL) R>=3, L=#evts')
    json_out     = Output('Path of the output file containing the results of the comparison')

    max_dt       = IntegerParameter(description='Tolerance for matching events (in timepoints)', optional=True, default=20)

    def run(self):
        # for now simply wrap the existing function
        from ml_ephys.validation.p_compare_ground_truth import compare_ground_truth as original_compare_ground_truth
        original_compare_ground_truth(
            firings_true=self.firings_true,
            firings=self.firings,
            json_out=self.json_out,
            max_dt=self.max_dt
        )

@register_processor(registry)
class synthesize_timeseries(Processor):

    VERSION="0.11a"

    firings = MdaInput('The path of firings events file in .mda format.', optional=True)
    waveforms = MdaInput('The path of (possibly upsampled) waveforms file in .mda format.', optional=True)

    timeseries_out = Output('The output path for the new timeseries. MxN')

    noise_level = FloatParameter('Standard deviation of the simulated background noise added to the timeseries', optional=True, default=1)
    samplerate  = FloatParameter('Sample rate for the synthetic dataset in seconds. The number of timepoints will be duration*samplerate', optional=True, min=0, default=30000)
    duration    = FloatParameter('Duration of the synthetic dataset in seconds. The number of timepoints will be duration*samplerate', optional=True, min=0, default=60)
    waveform_upsamplefac = IntegerParameter('The upsampling factor corresponding to the input waveforms. (avoids digitization artifacts)')
    amplitudes_row = IntegerParameter('If positive, this is the row in the firings arrays where the amplitude scale factors are found. Otherwise, use all 1\'s', optional=True, default=0)

    def run(self):
        from ml_ephys.synthesis.p_synthesize_timeseries import synthesize_timeseries as original_synthesize_timeseries
        original_synthesize_timeseries(
            firings=self.firings or '',
            waveforms=self.waveforms or '',
            timeseries_out=self.timeseries_out,
            noise_level=self.noise_level,
            samplerate=self.samplerate,
            duration=self.duration,
            waveform_upsamplefac=self.waveform_upsamplefac,
            amplitudes_row=self.amplitudes_row
        )

    def test():
        from ml_ephys.synthetis.p_synthetize_timeseries import test_synthetize_timeseries
        return test_synthetize_timeseries()

@register_processor(registry)
class synthesize_random_firings(Processor):
    VERSION='0.14'
    firings_out = Output('Path to output firings mda file. 3xL, L is the number of events, second row are timestamps, third row are integer unit labels')
    K = IntegerParameter('Number of simulated units', optional=True, default=20)
    samplerate = FloatParameter('Sampling frequency in Hz', min=0, optional=True, default=30000)
    duration = FloatParameter('Duration of the simulated acquisition in seconds', optional=True, default=60)

    def run(self):
        from ml_ephys.synthesis.p_synthesize_random_firings import synthesize_random_firings as original_synthesize_random_firings
        original_synthesize_random_firings(
            firings_out = self.firings_out,
            K = self.K,
            samplerate = self.samplerate,
            duration = self.duration
        )

    @classmethod
    def test(cls):
        print('Running test')
        from ml_ephys.synthesis.p_synthesize_random_firings import test_synthesize_random_firings
        return test_synthesize_random_firings()

@register_processor(registry)
class synthesize_random_waveforms(Processor):
    VERSION='0.1'
    waveforms_out = Output('Path to waveforms mda file. Mx(T*upsamplefac)xK')
    geometry_out = Output('Path to geometry csv file', optional=True)
    M = IntegerParameter('Number of channels', optional=True, default=5, min=1)
    T = IntegerParameter('Number of timepoints for a waveform, before upsampling', default=500, optional=True, min=1)
    K = IntegerParameter('Number of waveforms to synthesize', optional=True, default=20)
    timeshift_factor = IntegerParameter('Controls amount of timeshift between waveforms on different channels for each template', optional=True, default=3)
    upsample_fac = IntegerParameter('used for upsampling the waveforms to avoid discretization artifacts', optional=True, default=13)
    average_peak_amplitude = FloatParameter('used to scale the peak spike amplitude', optional=True, default=10)

    def run(self):
        from ml_ephys.synthesis.p_synthesize_random_waveforms import synthesize_random_waveforms as original_synthesize_random_waveforms
        original_synthesize_random_waveforms(
            waveforms_out=self.waveforms_out,
            geometry_out=self.geometry_out,
            M=self.M,
            T=self.T,
            K=self.K,
            timeshift_factor=self.timeshift_factor,
            upsamplefac=self.upsamplefac,
            average_peak_amplitude=self.average_peak_amplitude
        )

    def test():
        from ml_ephys.synthesis.p_synthesize_random_waveforms import test_synthesize_random_waveforms
        return test_synthesize_random_waveforms()

#@register_processor(registry)
class synthesize_drifting_timeseries(Processor):
    VERSION='0.1'

    firings = MdaInput('The path of firing events file in .mda format. RxL where '
    'R>=3 and L is the number of events. Second row is the timestamps, '
    'third row is the integer labels', optional=True)
    waveforms = MdaInput('The path of (possibly upsampled) waveforms file in .mda '
    'format. Mx(T*waveform_upsample_factor)*(2K), where M is the number of '
    'channels, T is the clip size, and K is the number of units. Each unit '
    'has a contiguous pair of waveforms (interpolates from first to second '
    'across the timeseries', optional=True)
    timeseries_out = Output('The output path for the new timeseries. MxN')

    noise_level = FloatParameter('Standard deviation of the simulated background noise added to the timeseries', optional=True, default=1)
    samplerate = FloatParameter('Sample rate for the synthetic dataset in Hz', optional=True, default=30000)
    duration = FloatParameter('Duration of the synthetic dataset in seconds. The number of timepoints will be duration*samplerate', optional=True, default=60)
    waveform_upsamplefac = IntegerParameter('The upsampling factor corresponding to the input waveforms. (avoids digitization artifacts)', optional=True, default=1)
    amplitudes_row = IntegerParameter("If positive, this is the row in the firings arrays where the amplitude scale factors are found. Otherwise, use all 1's", optional=True, default=0)
    num_interp_nodes = IntegerParameter('For drift, the number of timepoints where we specify the waveform', optional=True, default=2)

    def run(self):
        from ml_ephys.synthesis.p_synthesize_drifting_timeseries import synthesize_drifting_timeseries as original_synthesize_drifting_timeseries
        original_synthesize_drifting_timeseries(
            firings = self.firings,
            waveforms = self.waveforms,
            timeseries_out = self.timeseries_out,
            noise_level = self.noise_level,
            samplerate = self.samplerate,
            duration = self.duration,
            waveform_upsamplefac = self.waveform_upsamplefac,
            amplitudes_row = self.amplitudes_row,
            num_interp_nodes = self.num_interp_nodes
        )

    def test():
        from ml_ephys.synthesis.p_synthesize_drifting_timeseries import test_synthesize_drifting_timeseries
        return test_synthesize_drifting_timeseries()

@register_processor(registry)
class bandpass_filter(Processor):
    VERSION='0.1'

    timeseries = Input('MxN raw timeseries array (M = #channels, N = #timepoints)')
    timeseries_out = Output('Filtered output (MxN array)')
    samplerate = FloatParameter('The sampling rate in Hz')
    freq_min = FloatParameter('The lower endpoint of the frequency band (Hz)')
    freq_max = FloatParameter('The upper endpoint of the frequency band (Hz)')
    freq_wid = FloatParameter('The optional width of the roll-off (Hz)', optional=True, default=1000)
    #padding = 3000,
    #chunk_size=3000*10,
    #num_processes=os.cpu_count()

    def run(self):
        from ml_ephys.preprocessing.p_bandpass_filter import bandpass_filter as original_bandpass_filter
        original_bandpass_filter(
            timeseries     = self.timeseries,
            timeseries_out = self.timeseries_out,
            samplerate     = self.samplerate,
            freq_min       = self.freq_min,
            freq_max       = self.freq_max,
            freq_wid       = self.freq_wid
        )

import sys

if __name__ == "__main__":
    registry.process(sys.argv)
