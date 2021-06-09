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
#from lstchain.reco import r0_to_dl1
#from lstchain.io.io import dl1_images_lstcam_key, dl1_params_tel_mon_ped_key, dl1_params_tel_mon_cal_key, dl1_params_lstcam_key, dl1_params_src_dep_lstcam_key
from scipy.stats import binned_statistic
from datetime import datetime
from logging import getLogger,StreamHandler,DEBUG,INFO,WARNING,ERROR,CRITICAL

##### Logger #####
logger = getLogger(__name__)
handler = StreamHandler()
loglevel = 'DEBUG'
handler.setLevel(loglevel)
logger.setLevel(loglevel)
logger.addHandler(handler)


class ImageExtracterConfig:
    def __init__(self, image_extractor, dict_parameters={}, output_dir_path=Path('.'), config_dir_path=Path('.')):

        # Config parameters
        self.image_extractor = image_extractor
        self.parameters = dict_parameters
        self.config_name = self.image_extractor
        self.abbreviation = self.config_name
        if self.image_extractor=='LocalPeakWindowSum':
            self.abbreviation = 'LPWSs{shift}w{width}c{correct}'.format(shift=dict_parameters['window_shift'], \
                                                                        width=dict_parameters['window_width'], \
                                                                        correct=dict_parameters['apply_integration_correction'].capitalize()[0])
        elif self.image_extractor=='NeighborPeakWindowSum':
            self.abbreviation = 'NPWSs{shift}w{width}l{lwt}c{correct}'.format(shift=dict_parameters['window_shift'], \
                                                                        width=dict_parameters['window_width'], \
                                                                        lwt=dict_parameters['lwt'], \
                                                                        correct=dict_parameters['apply_integration_correction'].capitalize()[0])
        elif self.image_extractor=='SlidingWindowMaxSum':
            self.abbreviation = 'SWMSw{width}c{correct}'.format(width=dict_parameters['window_width'], \
                                                                correct=dict_parameters['apply_integration_correction'].capitalize()[0])
        elif self.image_extractor=='TwoPassWindowSum':
            self.abbreviation = 'TPWSt{threshold}d{disable2nd}c{correct}'.format(threshold=dict_parameters['core_threshold'], \
                                                                                 disable2nd=dict_parameters['disable_second_pass'].capitalize()[0], \
                                                                                 correct=dict_parameters['apply_integration_correction'].capitalize()[0])
        elif self.image_extractor=='FullWaveformSum':
            self.abbreviation = 'FWS'
             
        else:
            self.abbreviation = self.config_name
       
        
        # Output file name and path
        for par_key, par_val in self.parameters.items():
            self.config_name += '_{0}{1}'.format(par_key, par_val)
        self.config_path = config_dir_path / '{0}.json'.format(self.config_name)
        self.product_dir_path = output_dir_path / self.config_name
        if not self.product_dir_path.exists():
            os.makedirs(self.product_dir_path)   
        self.dl1_file_path = None
        
        # Readout data
        self.dl1_data = None
        self.dl1_image_tables = None
        
        # Performance results
        self.dl1_reco_phe = None
        self.dl1_reco_log10phe = None        
        self.dl1_reco_true_frac = None
        self.dl1_reco_true_devfrac = None
        self.dl1_reco_average_phe = {}
        self.reco_stats = {'sum': None, 'mean': None, 'std': None, 'count': None}        
        self.reco_stats_log = {'sum': None, 'mean': None, 'std': None, 'count': None}
        self.reco_frac_stats = {'sum': None, 'mean': None, 'std': None, 'count': None}        
        self.reco_frac_stats_log = {'sum': None, 'mean': None, 'std': None, 'count': None}        
        self.reco_devfrac_stats = {'sum': None, 'mean': None, 'std': None, 'count': None}        
        self.reco_devfrac_stats_log = {'sum': None, 'mean': None, 'std': None, 'count': None}  
        self.reco_phe_hists = {}

        
    def write_configfile(self, \
                         basicconfig_path=Path('./lstchain_basic_config_for_ChargeExtractionStudy.json')):
        with basicconfig_path.open() as basicconfig_file:
            config_string = basicconfig_file.read()
            config_string = config_string.replace('__IMAGE_EXTRACTER__',
                                                  self.image_extractor)
            parameter_block_string = '"{0}":{{\n'.format(self.image_extractor)
            parameter_block_string += ',\n'.join(['    "{0}": {1}'.format(par_key, par_val) \
                                                  for par_key, par_val in self.parameters.items()])
                
            parameter_block_string += '\n  },'
            config_string = config_string.replace('__IMAGE_EXTRACTER_PARAMETER_BLOCK__',
                                                  parameter_block_string)
            logger.debug(parameter_block_string)
            logger.debug(config_string)
            with open(self.config_path, 'w') as thisconfig_file:
                thisconfig_file.write(config_string)
                
                
    def produce_mc_dl1(self, dl0_path='/fefs/aswg/data/mc/DL0/20200629_prod5_trans_80/gamma-diffuse/zenith_20deg/south_pointing/gamma_20deg_180deg_run1000___cta-prod5-lapalma_4LSTs_MAGIC_desert-2158m_mono_cone6.simtel.gz', script_path_str='/home/mitsunari.takahashi/Work/Soft/cta-lstchain/lstchain/scripts/lstchain_mc_r0_to_dl1.py'):
        self.dl1_file_name = '_'.join(['dl1', dl0_path.name.replace('.simtel.gz', '.h5')])        
        self.dl1_file_path = self.product_dir_path / self.dl1_file_name
        if not self.dl1_file_path.exists():
            dl1_production_result = subprocess.run(['python', script_path_str, '--input-file', '{dl0_path}'.format(dl0_path=dl0_path), '--config', '{config_path}'.format(config_path=self.config_path), '--output-dir', '{product_dir_path}'.format(product_dir_path=self.product_dir_path)], 
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            logger.info(dl1_production_result.stdout)
            logger.warning(dl1_production_result.stderr)
            return dl1_production_result.returncode
        else:
            dtfile = datetime.fromtimestamp(self.dl1_file_path.stat().st_mtime)
            dtnow = datetime.now()
            fileage = dtnow-dtfile
            logger.warning('{0} already exists! It was modified {1:.0f} hr {2:.0f} min before.'.format(self.dl1_file_path, fileage.seconds/3600, (fileage.seconds%3600)/60))
            return 0
            
            
    def read_dl1(self, event_used, tel_id=1): #, emin=10*u.GeV, emax=50*u.GeV):
        self.dl1_data = tables.open_file(self.dl1_file_path)
        images_tel = self.dl1_data.root.dl1.event.telescope.image.LST_LSTCam.where("""tel_id=={0}""".format(tel_id))
        self.dl1_image_tables = \
        [ x[2] for (x, y) in zip(images_tel, event_used) if y==True]
        #[ x[2] for x, y in zip(self.dl1_data.root.dl1.event.telescope.image.LST_LSTCam.where("""tel_id=={0} and mc_energy>={1} and mc_energy<{2}""".format(tel_id, emin.to(u.TeV).value, emax.to(u.TeV).value)), event_used)]
        self.dl1_entries = len(self.dl1_image_tables)
        logger.debug('DL1 event number: {0} events'.format(self.dl1_entries))
        dl1_reco_phe = [] 
        for dl1img in self.dl1_image_tables:
            for i in dl1img:
                dl1_reco_phe.append(i)
        self.dl1_reco_phe = np.array(dl1_reco_phe)    
        
    
    def calc_correlations(self, true_phe):
        true_zerophe = np.array(true_phe==0)
        true_phe_zeromasked = ma.masked_array(true_phe, mask=true_zerophe)
        self.dl1_reco_phe_truezeromasked = ma.masked_array(self.dl1_reco_phe, mask=true_zerophe)
        
        self.dl1_reco_true_frac = self.dl1_reco_phe_truezeromasked / true_phe_zeromasked
        self.dl1_reco_true_devfrac = self.dl1_reco_true_frac - 1.
        
        # Linear binning
        self.bins = np.linspace(0, 1000, 1001)
        for kstat in self.reco_stats.keys():
            self.reco_stats[kstat] = binned_statistic(true_phe, self.dl1_reco_phe, statistic=kstat, bins=self.bins)  
        self.mean_phe_true0 = self.reco_stats['mean'][0][0]
        for kstat in self.reco_frac_stats.keys():
            self.reco_frac_stats[kstat] = binned_statistic(true_phe, self.dl1_reco_true_frac, statistic=kstat, bins=self.bins)              
        for kstat in self.reco_devfrac_stats.keys():
            self.reco_devfrac_stats[kstat] = binned_statistic(true_phe, self.dl1_reco_true_devfrac, statistic=kstat, bins=self.bins)
            
        # Log binning
        self.logbins = np.logspace(0, 3, 16)
        for kstat in self.reco_stats_log.keys():
            self.reco_stats_log[kstat] = binned_statistic(true_phe_zeromasked, self.dl1_reco_phe, statistic=kstat, bins=self.logbins)
        for kstat in self.reco_frac_stats_log.keys():
            self.reco_frac_stats_log[kstat] = binned_statistic(true_phe_zeromasked, self.dl1_reco_true_frac, statistic=kstat, bins=self.logbins)
        for kstat in self.reco_devfrac_stats_log.keys():
            self.reco_devfrac_stats_log[kstat] = binned_statistic(true_phe_zeromasked, self.dl1_reco_true_devfrac, statistic=kstat, bins=self.logbins)
        

    def calc_separation(self, measure='entropy', true_phe_list=[0,3], ndivbin=0):
        measure_all = 0
        for div in (0, 1):
            nentries = {}
            nentries_total = 0
            for tphe in true_phe_list:
                nentries[tphe] = sum(self.reco_phe_hists[tphe][0][:ndivbin] if div==0 \
                                     else self.reco_phe_hists[tphe][0][ndivbin:]) / \
                sum(self.reco_phe_hists[tphe][0])
                nentries_total += nentries[tphe]
            if measure=='entropy':
                for tphe in true_phe_list:
                    p = nentries[tphe] / nentries_total if nentries_total>0 else 0                
                    measure_all += - p * math.log(p, len(true_phe_list)) if p>0 else 0
            elif measure=='gini':
                for tphe in true_phe_list:
                    p = nentries[tphe] / nentries_total if nentries_total>0 else 0
                    measure_all += p
            elif measure=='error':
                measure_all += 1 - max([nentries[tphe] / nentries_total if nentries_total>0 else 0 \
                                        for tphe in true_phe_list])
        return measure_all
        
        
    def find_best_separation(self, measure='entropy', true_phe_list=[0,3]):
        measure_before_separation = self.calc_separation(measure, true_phe_list=true_phe_list, ndivbin=0)
        separation_best_divbin = 0
        separation_best = measure_before_separation
        for devbin, divbinedge in enumerate(self.reco_phe_hists[true_phe_list[0]][1]):
            measure_after_separation = self.calc_separation(measure, true_phe_list=true_phe_list, ndivbin=devbin)
            if measure_after_separation<separation_best:
                separation_best_divbin = devbin
                separation_best = measure_after_separation
        separation_gain = measure_before_separation - separation_best
        logger.info('Best separation gain: {0} by {1}'.format(separation_gain, self.reco_phe_hists[true_phe_list[0]][1][separation_best_divbin]))
        return (separation_best_divbin, separation_gain)
    
    
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
