from neuron import h,gui
import pandas as pd
import os
import logging
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent.parent.parent.parent.absolute()))

from simulating_neurons.neuron_models.model_utils import create_synapses

logger = logging.getLogger(__name__)

# 0.13525 to get 1040 segs
# 0.13520 to get 1042 segs
def create_cell(path=None, max_segment_length=0.13520):
    import neuron
    h.load_file("import3d.hoc")
    h.load_file("nrngui.hoc")

    if path is None:
        path = os.path.dirname(os.path.realpath(__file__)) +'/'
    
    neuron.load_mechanisms(path)

    morphologyFilename = path + "/morphologies/cell1_only_soma.asc"
    biophysicalModelFilename = path + "/L5PCbiophys5b.hoc"
    biophysicalModelTemplateFilename = path + "/L5PCtemplate_2.hoc"

    h.load_file(biophysicalModelFilename)
    h.load_file(biophysicalModelTemplateFilename)
    cell = h.L5PCtemplate(morphologyFilename, max_segment_length)

    syn_df = create_synapses(cell, 'rat')

    logger.info(f"Created model with {len(syn_df['segments'])} segments")
    logger.info(f"Temperature is {h.celsius} degrees celsius")
    
    return cell, syn_df
