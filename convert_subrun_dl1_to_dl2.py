#!/usr/bin/env python3

import sys
import os
import os.path
import argparse
import subprocess
import pandas as pd
from pathlib import Path
import lstchain.reco.dl1_to_dl2 as reco
from lstchain.io.config import get_standard_config, read_configuration_file
from lstchain.reco.utils import filter_events
from lstchain.io.io import write_dataframe
from lstchain.io.io import dl1_params_lstcam_key, dl1_params_src_dep_lstcam_key, dl2_params_lstcam_key, dl2_params_src_dep_lstcam_key


parser = argparse.ArgumentParser(description="DL1 to DL2")

# Required arguments
parser.add_argument('--input-file', '-f', type=str,
                    dest='input_file',
                    help='path to a DL1 HDF5 file',
                    default=None, required=True)

parser.add_argument('--path-models', '-p', action='store', type=str,
                    dest='path_models',
                    help='Path where to find the trained RF',
                    default='./trained_models')

# Optional arguments
parser.add_argument('--output-file', '-o', action='store', type=str,
                    dest='output_file',
                    help='Path of the output DL2 HDF5 file',
                    default=None)

parser.add_argument('--config', '-c', action='store', type=str,
                    dest='config_file',
                    help='Path to a configuration file. If none is given, a standard configuration is applied',
                    default=None, required=False)


args = parser.parse_args()


def main():
    subrun_datapath = Path(args.input_file)
    if subrun_datapath.exists():
        print(f'reading subrun file : {subrun_datapath}')
        datapath_srcdep = Path(args.input_file.replace('dl1_','dl1_srcdep_'))
        if datapath_srcdep.exists():
            os.remove(datapath_srcdep)
        copy(subrun_datapath, datafile_srcdep)
        data_dl1 = pd.read_hdf(datafile_srcdep,key=dl1_params_lstcam_key)
        data_dl1_srcdep = pd.concat(reco.get_source_dependent_parameters(data_dl1, config), axis=1)
        write_dataframe(data_dl1_srcdep, datafile_srcdep, dl1_params_src_dep_lstcam_key)
        print(f'converting subrun DL1 file {subrun_datapath} to DL2.')
        !python ~/Work/Soft/cta-lstchain/lstchain/scripts/lstchain_dl1_to_dl2.py \
            --input-file {datafile_srcdep} \
            --path-models ~/Work/Exercise/LST1RealDataAnalysis/DL1analysis/model_srcdep/zenith_40deg \
            --output-dir ./zenith_40deg --config ../../DL1analysis/config_srcdep_wobble.json


if __name__ == '__main__':
    main()
