#!/usr/bin/env python3

import sys

from mountainlab_pytools import processormanager as pm

import p_bandpass_filter
import p_whiten

PM=pm.ProcessorManager()

PM.registerProcessor(p_bandpass_filter.bandpass_filter)
PM.registerProcessor(p_whiten.whiten)

if not PM.run(sys.argv):
    exit(-1)
