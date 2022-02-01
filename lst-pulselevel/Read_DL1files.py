#!/usr/bin/env python
import math
import numpy as np
import numpy.ma as ma
import os
import subprocess
from glob import glob
from pathlib import Path
import click
import tables
from astropy.io import fits
import astropy.units as u
from astropy.table import Table, vstack
#from scipy.stats import binned_statistic
from datetime import datetime
import pickle
from logging import getLogger,StreamHandler,DEBUG,INFO,WARNING,ERROR,CRITICAL

#from traitlets.config.loader import Config
#from ctapipe_io_lst import LSTEventSource
#from ctapipe.instrument import CameraGeometry
#from ctapipe.visualization import CameraDisplay
#from ctapipe.image import hillas_parameters

#from lstchain.io.config import read_configuration_file
#import lstchain.reco.utils as utils
#from lstchain.reco import r0_to_dl1
#from lstchain.io.io import dl1_images_lstcam_key, dl1_params_tel_mon_ped_key, dl1_params_tel_mon_cal_key, dl1_params_lstcam_key, dl1_params_src_dep_lstcam_key

#from ctapipe.utils import get_dataset_path
#from ctapipe.io import EventSource
#from ctapipe.io.eventseeker import EventSeeker

from LowLevelData import Data, DL0Data, DL0DataSingleFile, DL1Data, DL1DataSingleFile


##### Logger #####
logger = getLogger(__name__)
handler = StreamHandler()
loglevel = 'INFO'
handler.setLevel(loglevel)
logger.setLevel(loglevel)
logger.addHandler(handler)


def read_dl1_subrun_files(run, datasrc, name, subrunpaths, parameters, emin, emax, tel):
    dl1_data = DL1Data(name=name, datasrc=datasrc, 
                       hillas_parameters=parameters, 
                       emin=emin, emax=emax, tel_id=tel)  
    for isubrun, subrunpath in enumerate(subrunpaths):
        subrun_dl1_path = Path(subrunpath)
        subrun_data_list = []
        if subrun_dl1_path.is_file(): 
            logger.info('{0} exists.'.format(subrun_dl1_path))
                #continue
        else:             
            logger.error('{0} has NOT been produced!!'.format(subrun_dl1_path))
            break
        subrun_data_list.append(DL1DataSingleFile(name='{0}-{1}'.format(name, isubrun), 
                                                  file_path=subrun_dl1_path, datasrc=datasrc, used_event_ids=None, 
                                                  true_phe=None, true_log10_phe=None, 
                                                  hillas_parameters=parameters, 
                                                  emin=emin, emax=emax, tel_id=tel))
        dl1_data.add_data(datum_list=subrun_data_list)
    return dl1_data


@click.command()
@click.argument('run', type=int)
@click.option('--name', '-n', type=str, default=None)
@click.option('--subrunpaths', '-s', type=str, multiple=True)
@click.option('--datasrc', '-d', type=click.Choice(['MC', 'Real']), default='Real')
@click.option('--parameters', '-p', type=str, multiple=True)
@click.option('--emin', type=float, default=0, help='Min energy in GeV')
@click.option('--emax', type=float, default=1000000, help='Min energy in GeV')
@click.option('--tel', type=int, default=1)
@click.option('--picklepath', type=str, default='./DL1Data.pickle')
@click.option('--loglevel', type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'CRITICAL']), default='INFO')
def main(run, name, subrunpaths, datasrc, parameters, emin, emax, tel, picklepath, loglevel):
    ##### Logger #####
    handler.setLevel(loglevel)
    logger.setLevel(loglevel)
    logger.addHandler(handler)

    if name is None:
        name = 'DL1 {0} Run {1}'.format(datasrc, run)

    dl1_data = read_dl1_subrun_files(run=run, datasrc=datasrc, name=name, subrunpaths=subrunpaths, parameters=parameters, emin=emin, emax=emax, tel=tel)
    with open(picklepath, 'wb') as f:
        pickle.dump(dl1_data, f)


if __name__ == '__main__':
    main()
