#!/usr/bin/env python
import math
import numpy as np
import numpy.ma as ma
import os
import subprocess
from glob import glob
from pathlib import Path
import tables
from astropy.io import fits
import astropy.units as u
from astropy.table import Table, vstack
#from scipy.stats import binned_statistic
from datetime import datetime
from logging import getLogger,StreamHandler,DEBUG,INFO,WARNING,ERROR,CRITICAL

from traitlets.config.loader import Config
from ctapipe_io_lst import LSTEventSource
from ctapipe.instrument import CameraGeometry
from ctapipe.visualization import CameraDisplay
from ctapipe.image import hillas_parameters

from lstchain.io.config import read_configuration_file
import lstchain.reco.utils as utils
from lstchain.reco import r0_to_dl1
from lstchain.io.io import dl1_images_lstcam_key, dl1_params_tel_mon_ped_key, dl1_params_tel_mon_cal_key, dl1_params_lstcam_key, dl1_params_src_dep_lstcam_key

from ctapipe.utils import get_dataset_path
from ctapipe.io import EventSource
from ctapipe.io.eventseeker import EventSeeker


##### Logger #####
logger = getLogger(__name__)
handler = StreamHandler()
loglevel = 'DEBUG'
handler.setLevel(loglevel)
logger.setLevel(loglevel)
logger.addHandler(handler)


class Data:
    def __init__(self, name, datasrc, emin=10*u.GeV, emax=50*u.GeV, tel_id=1):
        self.name = name
        self.file_paths = []
        self.DATASRC = datasrc
        self.TEL_ID = tel_id
        self.EMIN = emin
        self.EMAX = emax


class DL0Data(Data):
    def __init__(self, name, tel_id=1):
        Data.__init__(self, name=name, datasrc='MC', tel_id=tel_id)
        self.true_phe = np.ndarray([], dtype=float)
        self.true_log10_phe = np.ndarray([], dtype=float)
        self.event_used = np.ndarray([], dtype=bool)


    def add_files(self, file_paths):
        logger.info('Adding {0}...'.format(file_paths))
        counter = 0
        for file_path in file_paths:
            true_phe = []
            true_log10_phe = []
            true_energy = []
            event_used = []
            self.file_paths.append(file_path)

            source = EventSource(file_path)
            source.allowed_tels = [self.TEL_ID]
            for event in source:
                event_id = event.index.event_id
                true_image = event.simulation.tel[self.TEL_ID].true_image
                # Energy cut
                if self.EMIN <= event.simulation.shower.energy < self.EMAX:
                    event_used.append(True)
                    for trimg in true_image:
                        true_phe.append(trimg)
                        true_log10_phe.append(np.log10(max(0.1,trimg)))
                    counter+=1
                else:
                    event_used.append(False)

            np.append(self.true_phe, true_phe)
            np.append(self.true_log10_phe, true_log10_phe)
            np.append(self.event_used, event_used)

        logger.info('{0} events has been added.'.format(counter))


class DL1Data(Data):
    def __init__(self, name, datasrc, hillas_parameters=['intensity', 'length', 'width'], tel_id=1):
        Data.__init__(self, name=name, datasrc=datasrc, tel_id=tel_id)

        self.parameter_value_dict = {}
        for param in hillas_parameters:
            self.parameter_value_dict[param] = np.ndarray([], dtype=bool)
            #parameters_tel = self.dl1_data.root.dl1.event.telescope.parameters.LST_LSTCam.where("""tel_id=={0}""".format(tel_id))
            #self.parameter_value_dict[param] = np.array([x[param] for (x, y) in zip(parameters_tel, event_used) if y==True])
            #logger.debug('DL1 parameter-set number: {0} events'.format(len(self.parameter_value_dict[param])))
        
        self.dl1_image_tables = [] # Charge at every pixel for each event
        self.dl1_entries = len(self.dl1_image_tables)
        logger.debug('DL1 image number: {0} events'.format(self.dl1_entries))
        self.dl1_reco_phe = np.ndarray([], dtype=bool)


    def add_files(self, file_paths):
