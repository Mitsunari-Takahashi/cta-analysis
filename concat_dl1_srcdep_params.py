#!/usr/bin/env python3

import sys
import os
import argparse
import pandas as pd
from pathlib import Path
import lstchain.reco.dl1_to_dl2 as reco
from lstchain.io.config import get_standard_config, read_configuration_file
from lstchain.io.io import write_dataframe
from lstchain.io.io import dl1_params_lstcam_key, dl1_params_src_dep_lstcam_key, dl2_params_lstcam_key, dl2_params_src_dep_lstcam_key


parser = argparse.ArgumentParser(description="DL1 to DL2")

# Required arguments
parser.add_argument('--input-file', '-f', type=str,
                    dest='input_file',
                    help='path to a DL1 HDF5 file',
                    default=None, required=True)

# Optional arguments
parser.add_argument('--config', '-c', action='store', type=str,
                    dest='config_file',
                    help='Path to a configuration file. If none is given, a standard configuration is applied',
                    default=None, required=False)

args = parser.parse_args()


def main():
    datapath = Path(args.input_file)
    print('Reading DL1 file : {data}'.format(data=datapath))
    data_dl1 = pd.read_hdf(datapath,key=dl1_params_lstcam_key)

    if args.config_file==None:
        print('Standard config is used.')
        config = get_standard_config()
    else:
        print('Reading config file {0}'.format(args.config_file))
        config = read_configuration_file(args.config_file)

    data_dl1_srcdep = pd.concat(reco.get_source_dependent_parameters(data_dl1, config), axis=1)
    write_dataframe(data_dl1_srcdep, datapath, dl1_params_src_dep_lstcam_key)


if __name__ == '__main__':
    main()
