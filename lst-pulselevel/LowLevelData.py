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
from scipy.stats import binned_statistic
from datetime import datetime
import pickle
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
loglevel = 'INFO'
handler.setLevel(loglevel)
logger.setLevel(loglevel)
logger.addHandler(handler)


class Data:
    def __init__(self, name, datasrc, file_paths={}, emin=10*u.GeV, emax=50*u.GeV, tel_id=1):
        self.name = name
        self.file_paths = file_paths
        self.DATASRC = datasrc
        self.TEL_ID = tel_id
        self.EMIN = emin
        self.EMAX = emax


class DL0Data(Data):
    def __init__(self, name, file_paths={}, emin=10*u.GeV, emax=50*u.GeV, tel_id=1):
        Data.__init__(self, name=name, datasrc='MC', file_paths=file_paths, emin=emin, emax=emax, tel_id=tel_id)
        self.name_list = []
        self.true_phe = np.array([], dtype=float)
        self.true_log10_phe = np.array([], dtype=float)
        #self.event_used = np.array([], dtype=bool)
        self.dl0_entries = 0


    def add_data(self, datum_list=[]):
        for datum in datum_list:
            self.file_paths.update(datum.file_paths)
          #   self.true_phe += datum.true_phe
            # self.true_log10_phe += datum.true_log10_phe
            # self.event_used += datum.event_used
            self.true_phe = np.append(self.true_phe, datum.true_phe)
            self.true_log10_phe = np.append(self.true_log10_phe, datum.true_log10_phe)
            #self.event_used = np.append(self.event_used, datum.event_used)
            self.dl0_entries += datum.dl0_entries
            if datum.EMIN>self.EMIN:
                self.EMIN = datum.EMIN
            if datum.EMAX<self.EMAX:
                self.EMAX = datum.EMAX


class DL0DataSingleFile(DL0Data):
    def __init__(self, name, file_path, emin=10*u.GeV, emax=50*u.GeV, tel_id=1):
        DL0Data.__init__(self, name=name, file_paths={name:file_path}, emin=emin, emax=emax, tel_id=tel_id)
        self.file_path = self.file_paths[self.name]
        counter = 0
        true_phe = []
        true_log10_phe = []
        true_energy = []
        #event_used = []
        used_event_ids = []

        source = EventSource(self.file_path)
        source.allowed_tels = [self.TEL_ID]
        for event in source:
            obs_id = event.index.obs_id
            event_id = event.index.event_id
            #print(event.simulation.tel[self.TEL_ID])
            true_image = event.simulation.tel[self.TEL_ID].true_image
            if isinstance(true_image, np.ndarray):
                # Energy cut
                if self.EMIN <= event.simulation.shower.energy < self.EMAX:
                    #event_used.append(True)
                    used_event_ids.append((obs_id, event_id))
                    for trimg in true_image:
                        true_phe.append(trimg)
                        true_log10_phe.append(np.log10(max(0.1,trimg)))
                    counter+=1
            #     else:
            #         event_used.append(False)
            # else:
            #     event_used.append(False)

        self.true_phe = np.array(true_phe)
        self.true_log10_phe = np.array(true_log10_phe)
        #self.event_used = np.array(event_used, dtype=bool)
        self.used_event_ids = used_event_ids
        self.dl0_entries = counter
        logger.info('{0} events has been added.'.format(counter))
        logger.debug('True phe: {0} pixels'.format(self.true_phe.size))
        #logger.debug('Events used: {0} counts'.format(sum(self.event_used)))


    def produce_mc_dl1(self, lowlevel_config, hillas_parameters=['intensity', 'length', 'width'], script_path_str='/home/mitsunari.takahashi/Work/Soft/cta-lstchain/lstchain/scripts/lstchain_mc_r0_to_dl1.py'):
        dl1_production_result = None
        dl1_file_name = '_'.join(['dl1', self.file_path.name.replace('.simtel.gz', '.h5')])
        dl1_dir_path = lowlevel_config.product_dir_path / 'mc' / 'DL1'
        if not dl1_dir_path.is_dir():
            os.makedirs(dl1_dir_path)             
        dl1_file_path = dl1_dir_path / dl1_file_name

        if not dl1_file_path.exists():
            dl1_production_result = subprocess.run(['srun', 'python', script_path_str, '--input-file', '{dl0_path}'.format(dl0_path=self.file_path), '--config', '{config_path}'.format(config_path=lowlevel_config.config_path), '--output-dir', '{product_dir_path}'.format(product_dir_path=dl1_dir_path)], 
                                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            logger.info(dl1_production_result.stdout)
            logger.warning(dl1_production_result.stderr)
        else:
            dtfile = datetime.fromtimestamp(dl1_file_path.stat().st_mtime)
            dtnow = datetime.now()
            fileage = dtnow-dtfile
            logger.warning('{0} already exists! It was modified {1:.0f} days {2:.0f} hours {3:.0f} minutes before.'.format(dl1_file_path, fileage.days, fileage.seconds/3600, (fileage.seconds%3600)/60))

        dl1_datum = DL1DataSingleFile(name=self.name.replace('DL0', 'DL1').replace('dl0', 'dl1'), 
                                      file_path=dl1_file_path, 
                                      datasrc=self.DATASRC, 
                                      #event_used=self.event_used, 
                                      used_event_ids=self.used_event_ids,
                                      true_phe = self.true_phe,
                                      true_log10_phe = self.true_log10_phe,
                                      hillas_parameters=hillas_parameters,
                                      emin=self.EMIN, emax=self.EMAX) 

        return dl1_datum


class DL1Data(Data):
    def __init__(self, name, datasrc, file_paths={}, hillas_parameters=['intensity', 'length', 'width'], emin=10*u.GeV, emax=50*u.GeV, tel_id=1):
        Data.__init__(self, name=name, datasrc=datasrc, file_paths=file_paths, emin=emin, emax=emax, tel_id=tel_id)
        
        self.showerlevel_mask_dict = {}
        self.parameter_value_dict = {}
        self.hillas_parameters = hillas_parameters
        for param in self.hillas_parameters:
            self.parameter_value_dict[param] = [] #np.array([])
        
        self.dl1_image_tables = [] # Charge at every pixel for each event
        self.dl1_entries = len(self.dl1_image_tables)
        self.dl1_reco_phe = np.array([], dtype=bool)
        self.true_phe = np.array([], dtype=float)
        self.true_log10_phe = np.array([], dtype=float)

        self.reco_phe_stats = {'sum': None, 'mean': None, 'std': None, 'count': None}        
        self.reco_phe_stats_log = {'sum': None, 'mean': None, 'std': None, 'count': None}
        self.reco_phe_frac_stats = {'sum': None, 'mean': None, 'std': None, 'count': None}        
        self.reco_phe_frac_stats_log = {'sum': None, 'mean': None, 'std': None, 'count': None}        
        self.reco_phe_devfrac_stats = {'sum': None, 'mean': None, 'std': None, 'count': None}        
        self.reco_phe_devfrac_stats_log = {'sum': None, 'mean': None, 'std': None, 'count': None}  
        self.reco_phe_hists = {}


    def get_showerlevel_mask_sum(self):
        mask_sum = np.zeros_like(self.dl1_reco_phe)
        for mask in self.showerlevel_mask_dict.values():
            mask_sum += mask
        return mask_sum
    
        
    def add_data(self, datum_list=[]):
        logger.info('Adding {0}...'.format(datum_list))

        for datum in datum_list:
            self.file_paths.update(datum.file_paths)
            self.dl1_entries += datum.dl1_entries
            self.dl1_image_tables += datum.dl1_image_tables
            if self.DATASRC=='MC': #datum.true_phe!=None:
                self.true_phe = np.append(self.true_phe, datum.true_phe)
            if self.DATASRC=='MC': #datum.true_log10_phe!=None:
                self.true_log10_phe = np.append(self.true_log10_phe, datum.true_log10_phe)

            self.dl1_reco_phe = np.append(self.dl1_reco_phe, datum.dl1_reco_phe)

            if datum.EMIN>self.EMIN:
                self.EMIN =  datum.EMIN
            if datum.EMAX<self.EMAX:
                self.EMAX = datum.EMAX
            for param in self.hillas_parameters:
                self.parameter_value_dict[param] += datum.parameter_value_dict[param]


    def make_hist_pulselevel(self, stats=['mean', 'std', 'count']):
        true_zerophe = self.true_phe==0
        true_phe_zeromasked = ma.masked_array(self.true_phe, mask=true_zerophe)
        self.dl1_reco_phe_truezeromasked = ma.masked_array(self.dl1_reco_phe, mask=true_zerophe)
        
        self.dl1_reco_true_phe_frac = self.dl1_reco_phe_truezeromasked / true_phe_zeromasked
        self.dl1_reco_true_phe_devfrac = self.dl1_reco_true_phe_frac - 1.
        
        # Linear binning
        bins = np.linspace(0, 1000, 1001)
        for kstat in stats:
            self.reco_phe_stats[kstat] = binned_statistic(self.true_phe, self.dl1_reco_phe, statistic=kstat, bins=bins)  
        self.mean_phe_true0 = self.reco_phe_stats['mean'][0][0]
        for kstat in stats:
            self.reco_phe_frac_stats[kstat] = binned_statistic(self.true_phe, self.dl1_reco_true_phe_frac, statistic=kstat, bins=bins)              
        for kstat in stats:
            self.reco_phe_devfrac_stats[kstat] = binned_statistic(self.true_phe, self.dl1_reco_true_phe_devfrac, statistic=kstat, bins=bins)
            
        # Log binning
        logbins = np.logspace(0, 3, 16)
        for kstat in stats:
            self.reco_phe_stats_log[kstat] = binned_statistic(true_phe_zeromasked, self.dl1_reco_phe, statistic=kstat, bins=logbins)
        for kstat in stats:
            self.reco_phe_frac_stats_log[kstat] = binned_statistic(true_phe_zeromasked, self.dl1_reco_true_phe_frac, statistic=kstat, bins=logbins)
        for kstat in stats:
            self.reco_phe_devfrac_stats_log[kstat] = binned_statistic(true_phe_zeromasked, self.dl1_reco_true_phe_devfrac, statistic=kstat, bins=logbins)



    def get_roc_curve(self, sig_phe=3, bkg_phe=0):
        signal_total = np.sum(self.reco_phe_hists[sig_phe][0])
        background_total = np.sum(self.reco_phe_hists[bkg_phe][0])
        signal_cum = np.cumsum(self.reco_phe_hists[sig_phe][0][::-1])[::-1] 
        background_cum = np.cumsum(self.reco_phe_hists[bkg_phe][0][::-1])[::-1] 
        
        sig_acceptance = []
        bkg_residual = []
        for ibin, (sigcum, bkgcum) in enumerate(zip(signal_cum, background_cum)):
            sig_acceptance.append(sigcum/signal_total)
            bkg_residual.append(bkgcum/background_total)
        return (np.array(sig_acceptance), np.array(bkg_residual))
 

class DL1DataSingleFile(DL1Data):
    def __init__(self, name, file_path, datasrc, used_event_ids=None, true_phe=None, true_log10_phe=None, hillas_parameters=['intensity', 'length', 'width'], emin=10*u.GeV, emax=50*u.GeV, tel_id=1):
        DL1Data.__init__(self, name=name, datasrc=datasrc, file_paths={name:file_path}, hillas_parameters=hillas_parameters, emin=emin, emax=emax, tel_id=tel_id)
        self.parameter_value_dict = {}

        self.used_event_ids = used_event_ids
        self.true_phe = true_phe
        self.true_log10_phe = true_log10_phe
        self.file_path = self.file_paths[self.name]
        self.dl1_data = tables.open_file(self.file_path)
        images_tel = self.dl1_data.root.dl1.event.telescope.image.LST_LSTCam#.where("""tel_id=={0}""".format(tel_id))

        for param in self.hillas_parameters:
            parameters_tel = self.dl1_data.root.dl1.event.telescope.parameters.LST_LSTCam#.where("""tel_id=={0}""".format(tel_id))
            self.parameter_value_dict[param] = []
            for param_tel in parameters_tel:
                if (self.used_event_ids==None or (param_tel[0], param_tel[1]) in self.used_event_ids) and param_tel["tel_id"]==tel_id:
                    self.parameter_value_dict[param].append(param_tel[param])

            logger.debug('DL1 parameter-set number: {0} events'.format(len(self.parameter_value_dict[param])))
        
        self.dl1_image_tables = []
        for image_tel in images_tel:
            if (self.used_event_ids==None or (image_tel[0], image_tel[1]) in self.used_event_ids) and image_tel["tel_id"]==tel_id:
                self.dl1_image_tables.append(image_tel[2])

        self.dl1_entries = len(self.dl1_image_tables)
        logger.debug('DL1 image number: {0} events'.format(self.dl1_entries))
        dl1_reco_phe = [] 
        for dl1img in self.dl1_image_tables:
            for i in dl1img:
                dl1_reco_phe.append(i)
        self.dl1_reco_phe = np.array(dl1_reco_phe)
        logger.debug('DL1 reconstructed pulses: {0} counts'.format(self.dl1_reco_phe))
