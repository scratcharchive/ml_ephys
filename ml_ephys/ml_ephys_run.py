#!/usr/bin/env python3

from mountainlab_pytools.mlprocessors.registry import registry, register_processor

registry.namespace = "ephys_new"


from validation.p_compare_ground_truth import compare_ground_truth as original_compare_ground_truth

from mountainlab_pytools.mlprocessors.core import Input, Output, Processor, IntegerParameter, FloatParameter
from mountainlab_pytools.mlprocessors.validators import FileExtensionValidator

class MdaInput(Input):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.validators.insert(0, FileExtensionValidator(["mda"]))


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
        original_compare_ground_truth(firings_true=self.firings_true, firings=self.firings, json_out=self.json_out, max_dt=self.max_dt)

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
        original_synthesize_timeseries(firings=self.firings or '', waveforms=self.waveforms or '', timeseries_out=self.timeseries_out,
                                       noise_level=self.noise_level, samplerate=self.samplerate, duration=self.duration,
                                       waveform_upsamplefac=self.waveform_upsamplefac, amplitudes_row=self.amplitudes_row)

    def test(self):
        from ml_ephys.synthetis.p_synthetize_timeseries import test_synthetize_timeseries
        test_synthetize_timeseries()

@register_processor(registry)
class synthesize_random_firings(Processor):
    VERSION='0.14'
    firings_out = Output('Path to output firings mda file. 3xL, L is the number of events, second row are timestamps, third row are integer unit labels')
    K = IntegerParameter('Number of simulated units', optional=True, default=20)
    samplerate = FloatParameter('Sampling frequency in Hz', min=0, optional=True, default=30000)
    duration = FloatParameter('Duration of the simulated acquisition in seconds', optional=True, default=60)

    def run(self):
        from ml_ephys.synthesis.p_synthesize_random_firings import synthesize_random_firings as original_synthesize_random_firings
        original_synthesize_random_firings(firings_out = self.firings_out, K = self.K, samplerate = self.samplerate, duration = self.duration)

    def test(self):
        from ml_ephys.synthesis.p_synthesize_random_firings import test_synthesize_random_firings
        test_synthesize_random_firings()

import sys

if __name__ == "__main__":
    registry.process(sys.argv)
