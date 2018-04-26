import sys

from mltools import processormanager as pm

import p_compute_templates

PM=pm.ProcessorManager()

PM.registerProcessor(p_compute_templates.compute_templates)

if not PM.run(sys.argv):
    exit(-1)
