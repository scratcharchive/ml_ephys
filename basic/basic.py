import sys

from mltools import processormanager as pm

import p_compute_templates
import p_compute_cross_correlograms

PM=pm.ProcessorManager()

PM.registerProcessor(p_compute_templates.compute_templates)
PM.registerProcessor(p_compute_cross_correlograms.compute_cross_correlograms)

if not PM.run(sys.argv):
    exit(-1)
