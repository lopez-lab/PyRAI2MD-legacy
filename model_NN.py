## Neural Networks interface for PyRAIMD
## Jingbai Li Jul 0 2020

import time,datetime,json
import numpy as np
from nn_pes import NeuralNetPes
from nn_pes_device import set_gpu

class DNN:
    ## This is the interface to GP

    def __init__(self,variables_all,id=None):
        ## data      : dict
        ##             All data from traning data
        ## pred_data : str
        ##             Filename for test set
        ## hyp_eg/nac: dict
        ##           : Hyperparameters for NNs
        ## x         : np.array
        ##             Inverse distance in shape of (batch,(atom*atom-1)/2)
        ## y_dict    : dict
        ##             Dictionary of y values for each model. Energy in Bohr, Gradients in Hatree/Bohr. Nac are unchanged.

        ## unpack variables
        set_gpu([]) #No GPU for prediction
        title                      = variables_all['control']['title']
        variables                  = variables_all['nn']
        data                       = variables['postdata']
        nn_eg_type                 = variables['nn_eg_type']
        nn_nac_type                = variables['nn_nac_type']
        hyp_eg                     = variables['eg'].copy()
        hyp_nac                    = variables['nac'].copy()
        hyp_eg2                    = variables['eg2'].copy()
        hyp_nac2                   = variables['nac2'].copy()

        ## setup hypers
        hyp_dict_eg  ={
                      'general'    :{
                                    'model_type'            : hyp_eg['model_type'],
                                    },
                      'model'      :{
                                    'atoms'                 : data['natom'],
                                    'states'                : data['nstate'], 
                                    'Depth'                 : hyp_eg['Depth'],
                                    'nn_size'               : hyp_eg['nn_size'],
                                    'use_invdist'           : True,
                                    'invd_mean'             : data['mean_invr'],
                                    'invd_std'              : data['std_invr'],
                                    'use_bond_angles'       : False,
                                    'angle_index'           : [],
                                    'angle_mean'            : 2,   
                                    'angle_std'             : 0.2,  
                                    'use_dihyd_angles'      : False,
                                    'dihyd_index'           : [],
                                    'dihyd_mean'            : -1,
                                    'dihyd_std'             : 2,  
                                    'activ'                 : hyp_eg['activ'],
                                    'activ_alpha'           : hyp_eg['activ_alpha'],
                                    'use_dropout'           : hyp_eg['use_dropout'],
                                    'dropout'               : hyp_eg['dropout'],
                                    'use_reg_activ'         : hyp_eg['use_reg_activ'],
                                    'use_reg_weight'        : hyp_eg['use_reg_weight'],
                                    'use_reg_bias'          : hyp_eg['use_reg_bias'],
                                    'reg_l1'                : hyp_eg['reg_l1'],
                                    'reg_l2'                : hyp_eg['reg_l2'],
                                    'loss_weights'          : hyp_eg['loss_weights'], 
                                    'y_energy_mean'         : data['mean_energy'],
                                    'y_energy_std'          : 1,                                   # data['std_energy']
                                    'y_energy_unit_conv'    : 27.21138624598853,                   # conversion Hatree to eV after scaling
                                    'y_gradient_unit_conv'  : 27.21138624598853/0.52917721090380,  # conversion from H/bohr to eV/A after scaling
                                    },
       	       	      'training'   :{
                                    'reinit_weights'        : hyp_eg['reinit_weights'],
                                    'val_disjoint'          : hyp_eg['val_disjoint'],
                                    'val_split'             : hyp_eg['val_split'],
                                    'epo'                   : hyp_eg['epo'],
                                    'epomin'                : hyp_eg['epomin'],
                                    'patience'              : hyp_eg['patience'],
                                    'max_time'              : hyp_eg['max_time'],
                                    'batch_size'            : hyp_eg['batch_size'],
                                    'delta_loss'            : hyp_eg['delta_loss'],
                                    'factor_lr'             : hyp_eg['factor_lr'],
                                    'epostep'               : hyp_eg['epostep'],
                                    'learning_rate_start'   : hyp_eg['learning_rate_start'],
                                    'learning_rate_stop'    : hyp_eg['learning_rate_stop' ],
                                    'learning_rate_step'    : hyp_eg['learning_rate_step'],
                                    'epoch_step_reduction'  : hyp_eg['epoch_step_reduction'],
                                    'use_linear_callback'   : False,
                                    'use_early_callback'    : False,
                                    'use_exp_callback'      : False,
                                    'use_step_callback'     : True,
       	       	                    },
       	       	      'retraining' :{
                                    'reinit_weights'        : hyp_eg['t_reinit_weights'],
                                    'val_disjoint'          : hyp_eg['t_val_disjoint'],
                                    'val_split'             : hyp_eg['t_val_split'],
                                    'epo'                   : hyp_eg['t_epo'],
                                    'epomin'                : hyp_eg['t_epomin'],
                                    'patience'              : hyp_eg['t_patience'],
                                    'max_time'              : hyp_eg['t_max_time'],
                                    'batch_size'            : hyp_eg['t_batch_size'],
                                    'delta_loss'            : hyp_eg['t_delta_loss'],
                                    'factor_lr'             : hyp_eg['t_factor_lr'],
                                    'epostep'               : hyp_eg['t_epostep'],
                                    'learning_rate_start'   : hyp_eg['t_learning_rate_start'],
                                    'learning_rate_stop'    : hyp_eg['t_learning_rate_stop' ],
                                    'learning_rate_step'    : hyp_eg['t_learning_rate_step'],
                                    'epoch_step_reduction'  : hyp_eg['t_epoch_step_reduction'],
                                    'use_linear_callback'   : False,
                                    'use_early_callback'    : False,
                                    'use_exp_callback'      : False,
                                    'use_step_callback'     : True,            
       	       	                    },
                      'resample'   :{
                                    'reinit_weights'        : hyp_eg['a_reinit_weights'],
                                    'val_disjoint'          : hyp_eg['a_val_disjoint'],
                                    'val_split'             : hyp_eg['a_val_split'],
                                    'epo'                   : hyp_eg['a_epo'],
                                    'epomin'                : hyp_eg['a_epomin'],
                                    'patience'              : hyp_eg['a_patience'],
                                    'max_time'              : hyp_eg['a_max_time'],
                                    'batch_size'            : hyp_eg['a_batch_size'],
                                    'delta_loss'            : hyp_eg['a_delta_loss'],
                                    'factor_lr'             : hyp_eg['a_factor_lr'],
                                    'epostep'               : hyp_eg['a_epostep'],
                                    'learning_rate_start'   : hyp_eg['a_learning_rate_start'],
                                    'learning_rate_stop'    : hyp_eg['a_learning_rate_stop' ],
                                    'learning_rate_step'    : hyp_eg['a_learning_rate_step'],
                                    'epoch_step_reduction'  : hyp_eg['a_epoch_step_reduction'],
                                    'use_linear_callback'   : False,
                                    'use_early_callback'    : False,
                                    'use_exp_callback'      : False,
                                    'use_step_callback'     : True, 	
       	       	                    },
                      }
        hyp_dict_nac ={
                      'general'    :{
                                    'model_type'            : hyp_nac['model_type'],
                                    },
                      'model'      :{
                                    'atoms'                 : data['natom'],
                                    'states'                : data['npair'], 
                                    'Depth'                 : hyp_nac['Depth'],
                                    'nn_size'               : hyp_nac['nn_size'],
                                    'use_invdist'           : True,
                                    'invd_mean'             : data['mean_invr'],
                                    'invd_std'              : data['std_invr'],
                                    'use_bond_angles'       : False,
                                    'angle_index'           : [],
                                    'angle_mean'            : 2,   
                                    'angle_std'             : 0.2,  
                                    'use_dihyd_angles'      : False,
                                    'dihyd_index'           : [],
                                    'dihyd_mean'            : -1,
                                    'dihyd_std'             : 2,
                                    'activ'                 : hyp_nac['activ'],
                                    'activ_alpha'           : hyp_nac['activ_alpha'],  
                                    'use_dropout'           : hyp_nac['use_dropout'],
                                    'dropout'               : hyp_nac['dropout'],
                                    'use_reg_activ'         : hyp_nac['use_reg_activ'],
                                    'use_reg_weight'        : hyp_nac['use_reg_weight'],
                                    'use_reg_bias'          : hyp_nac['use_reg_bias'],
                                    'reg_l1'                : hyp_nac['reg_l1'],
                                    'reg_l2'                : hyp_nac['reg_l2'],
                                    'y_nac_mean'            : 0,                  # data['mean_nac'],
                                    'y_nac_std'             : 1,                  #data['std_nac']
                                    'y_nac_unit_conv'       : 1/0.52917721090380, # conversion 1/Bohr to 1/A after scaling!!
                                    'phase_less_loss'       : hyp_nac['phase_less_loss'],
                                    },
       	       	      'training'   :{
                                    'phase_less_loss'       : hyp_nac['phase_less_loss'],
                                    'reinit_weights'        : hyp_nac['reinit_weights'],
                                    'val_disjoint'          : hyp_nac['val_disjoint'],
                                    'val_split'             : hyp_nac['val_split'],
                                    'epo'                   : hyp_nac['epo'],
                                    'pre_epo'               : hyp_nac['pre_epo'],
                                    'epomin'                : hyp_nac['epomin'],
                                    'patience'              : hyp_nac['patience'],
                                    'max_time'              : hyp_nac['max_time'],
                                    'batch_size'            : hyp_nac['batch_size'],
                                    'delta_loss'            : hyp_nac['delta_loss'],
                                    'factor_lr'             : hyp_nac['factor_lr'],
                                    'epostep'               : hyp_nac['epostep'],
                                    'learning_rate_start'   : hyp_nac['learning_rate_start'],
                                    'learning_rate_stop'    : hyp_nac['learning_rate_stop' ],
                                    'learning_rate_step'    : hyp_nac['learning_rate_step'],
                                    'epoch_step_reduction'  : hyp_nac['epoch_step_reduction'],
                                    'use_linear_callback'   : False,
                                    'use_early_callback'    : False,
                                    'use_exp_callback'      : False,
                                    'use_step_callback'     : True,
       	       	                    },
       	       	      'retraining' :{
                                    'phase_less_loss'       : hyp_nac['phase_less_loss'],
                                    'reinit_weights'        : hyp_nac['t_reinit_weights'],
                                    'val_disjoint'          : hyp_nac['t_val_disjoint'],
                                    'val_split'             : hyp_nac['t_val_split'],
                                    'epo'                   : hyp_nac['t_epo'],
                                    'pre_epo'               : hyp_nac['t_pre_epo'],
                                    'epomin'                : hyp_nac['t_epomin'],
                                    'patience'              : hyp_nac['t_patience'],
                                    'max_time'              : hyp_nac['t_max_time'],
                                    'batch_size'            : hyp_nac['t_batch_size'],
                                    'delta_loss'            : hyp_nac['t_delta_loss'],
                                    'factor_lr'             : hyp_nac['t_factor_lr'],
                                    'epostep'               : hyp_nac['t_epostep'],
                                    'learning_rate_start'   : hyp_nac['t_learning_rate_start'],
                                    'learning_rate_stop'    : hyp_nac['t_learning_rate_stop' ],
                                    'learning_rate_step'    : hyp_nac['t_learning_rate_step'],
                                    'epoch_step_reduction'  : hyp_nac['t_epoch_step_reduction'],
                                    'use_linear_callback'   : False,
                                    'use_early_callback'    : False,
                                    'use_exp_callback'      : False,
                                    'use_step_callback'     : True,            
       	       	                    },
                      'resample'   :{
                                    'phase_less_loss'       : hyp_nac['phase_less_loss'],
                                    'reinit_weights'        : hyp_nac['a_reinit_weights'],
                                    'val_disjoint'          : hyp_nac['a_val_disjoint'],
                                    'val_split'             : hyp_nac['a_val_split'],
                                    'epo'                   : hyp_nac['a_epo'],
                                    'pre_epo'               : hyp_nac['a_pre_epo'],
                                    'epomin'                : hyp_nac['a_epomin'],
                                    'patience'              : hyp_nac['a_patience'],
                                    'max_time'              : hyp_nac['a_max_time'],
                                    'batch_size'            : hyp_nac['a_batch_size'],
                                    'delta_loss'            : hyp_nac['a_delta_loss'],
                                    'factor_lr'             : hyp_nac['a_factor_lr'],
                                    'epostep'               : hyp_nac['a_epostep'],
                                    'learning_rate_start'   : hyp_nac['a_learning_rate_start'],
                                    'learning_rate_stop'    : hyp_nac['a_learning_rate_stop' ],
                                    'learning_rate_step'    : hyp_nac['a_learning_rate_step'],
                                    'epoch_step_reduction'  : hyp_nac['a_epoch_step_reduction'],
                                    'use_linear_callback'   : False,
                                    'use_early_callback'    : False,
                                    'use_exp_callback'      : False,
                                    'use_step_callback'     : True, 	
       	       	                    },
                      }
        hyp_dict_eg2 ={
                      'general'    :{
                                    'model_type'            : hyp_eg2['model_type'],
                                    },
                      'model'      :{
                                    'atoms'                 : data['natom'],
                                    'states'                : data['nstate'], 
                                    'Depth'                 : hyp_eg2['Depth'],
                                    'nn_size'               : hyp_eg2['nn_size'],
                                    'use_invdist'           : True,
                                    'invd_mean'             : data['mean_invr'],
                                    'invd_std'              : data['std_invr'],
                                    'use_bond_angles'       : False,
                                    'angle_index'           : [],
                                    'angle_mean'            : 2,   
                                    'angle_std'             : 0.2,  
                                    'use_dihyd_angles'      : False,
                                    'dihyd_index'           : [],
                                    'dihyd_mean'            : -1,
                                    'dihyd_std'             : 2,
                                    'activ'                 : hyp_eg2['activ'],
                                    'activ_alpha'           : hyp_eg2['activ_alpha'],
                                    'use_dropout'           : hyp_eg2['use_dropout'],
                                    'dropout'               : hyp_eg2['dropout'],
                                    'use_reg_activ'         : hyp_eg2['use_reg_activ'],
                                    'use_reg_weight'        : hyp_eg2['use_reg_weight'],
                                    'use_reg_bias'          : hyp_eg2['use_reg_bias'],
                                    'reg_l1'                : hyp_eg2['reg_l1'],
                                    'reg_l2'                : hyp_eg2['reg_l2'],
                                    'loss_weights'          : hyp_eg2['loss_weights'], 
                                    'y_energy_mean'         : data['mean_energy'],
                                    'y_energy_std'          : 1,                                   # data['std_energy']
                                    'y_energy_unit_conv'    : 27.21138624598853,                   # conversion Hatree to eV after scaling
                                    'y_gradient_unit_conv'  : 27.21138624598853/0.52917721090380,  # conversion from H/bohr to eV/A after scaling                                    },
                                    },
       	       	      'training'   :{
                                    'reinit_weights'        : hyp_eg2['reinit_weights'],
                                    'val_disjoint'          : hyp_eg2['val_disjoint'],
                                    'val_split'             : hyp_eg2['val_split'],
                                    'epo'                   : hyp_eg2['epo'],
                                    'epomin'                : hyp_eg2['epomin'],
                                    'patience'              : hyp_eg2['patience'],
                                    'max_time'              : hyp_eg2['max_time'],
                                    'batch_size'            : hyp_eg2['batch_size'],
                                    'delta_loss'            : hyp_eg2['delta_loss'],
                                    'factor_lr'             : hyp_eg2['factor_lr'],
                                    'epostep'               : hyp_eg2['epostep'],
                                    'learning_rate_start'   : hyp_eg2['learning_rate_start'],
                                    'learning_rate_stop'    : hyp_eg2['learning_rate_stop' ],
                                    'learning_rate_step'    : hyp_eg2['learning_rate_step'],
                                    'epoch_step_reduction'  : hyp_eg2['epoch_step_reduction'],
                                    'use_linear_callback'   : False,
                                    'use_early_callback'    : False,
                                    'use_exp_callback'      : False,
                                    'use_step_callback'     : True,
       	       	                    },
       	       	      'retraining' :{
                                    'reinit_weights'        : hyp_eg2['t_reinit_weights'],
                                    'val_disjoint'          : hyp_eg2['t_val_disjoint'],
                                    'val_split'             : hyp_eg2['t_val_split'],
                                    'epo'                   : hyp_eg2['t_epo'],
                                    'epomin'                : hyp_eg2['t_epomin'],
                                    'patience'              : hyp_eg2['t_patience'],
                                    'max_time'              : hyp_eg2['t_max_time'],
                                    'batch_size'            : hyp_eg2['t_batch_size'],
                                    'delta_loss'            : hyp_eg2['t_delta_loss'],
                                    'factor_lr'             : hyp_eg2['t_factor_lr'],
                                    'epostep'               : hyp_eg2['t_epostep'],
                                    'learning_rate_start'   : hyp_eg2['t_learning_rate_start'],
                                    'learning_rate_stop'    : hyp_eg2['t_learning_rate_stop' ],
                                    'learning_rate_step'    : hyp_eg2['t_learning_rate_step'],
                                    'epoch_step_reduction'  : hyp_eg2['t_epoch_step_reduction'],
                                    'use_linear_callback'   : False,
                                    'use_early_callback'    : False,
                                    'use_exp_callback'      : False,
                                    'use_step_callback'     : True,
       	       	                    },
                      'resample'   :{
                                    'reinit_weights'        : hyp_eg2['a_reinit_weights'],
                                    'val_disjoint'          : hyp_eg2['a_val_disjoint'],
                                    'val_split'             : hyp_eg2['a_val_split'],
                                    'epo'                   : hyp_eg2['a_epo'],
                                    'epomin'                : hyp_eg2['a_epomin'],
                                    'patience'              : hyp_eg2['a_patience'],
                                    'max_time'              : hyp_eg2['a_max_time'],
                                    'batch_size'            : hyp_eg2['a_batch_size'],
                                    'delta_loss'            : hyp_eg2['a_delta_loss'],
                                    'factor_lr'             : hyp_eg2['a_factor_lr'],
                                    'epostep'               : hyp_eg2['a_epostep'],
                                    'learning_rate_start'   : hyp_eg2['a_learning_rate_start'],
                                    'learning_rate_stop'    : hyp_eg2['a_learning_rate_stop' ],
                                    'learning_rate_step'    : hyp_eg2['a_learning_rate_step'],
                                    'epoch_step_reduction'  : hyp_eg2['a_epoch_step_reduction'],
                                    'use_linear_callback'   : False,
                                    'use_early_callback'    : False,
                                    'use_exp_callback'      : False,
                                    'use_step_callback'     : True,
       	       	                    },
                      }
        hyp_dict_nac2={
                      'general'    :{
                                    'model_type'            : hyp_nac2['model_type'],
                                    },
                      'model'      :{
                                    'atoms'                 : data['natom'],
                                    'states'                : data['npair'], 
                                    'Depth'                 : hyp_nac2['Depth'],
                                    'nn_size'               : hyp_nac2['nn_size'],
                                    'use_invdist'           : True,
                                    'invd_mean'             : data['mean_invr'],
                                    'invd_std'              : data['std_invr'],
                                    'use_bond_angles'       : False,
                                    'angle_index'           : [],
                                    'angle_mean'            : 2,   
                                    'angle_std'             : 0.2,  
                                    'use_dihyd_angles'      : False,
                                    'dihyd_index'           : [],
                                    'dihyd_mean'            : -1,
                                    'dihyd_std'             : 2,
                                    'activ'                 : hyp_nac2['activ'],
                                    'activ_alpha'           : hyp_nac2['activ_alpha'],
                                    'use_dropout'           : hyp_nac2['use_dropout'],
                                    'dropout'               : hyp_nac2['dropout'],
                                    'use_reg_activ'         : hyp_nac2['use_reg_activ'],
                                    'use_reg_weight'        : hyp_nac2['use_reg_weight'],
                                    'use_reg_bias'          : hyp_nac2['use_reg_bias'],
                                    'reg_l1'                : hyp_nac2['reg_l1'],
                                    'reg_l2'                : hyp_nac2['reg_l2'],
                                    'y_nac_mean'            : 0,                  # data['mean_nac'],
                                    'y_nac_std'             : 1,                  # data['std_nac']
                                    'y_nac_unit_conv'       : 1/0.52917721090380, # conversion 1/Bohr to 1/A after scaling!!
                                    'phase_less_loss'       : hyp_nac2['phase_less_loss'],
                                    },
       	       	      'training'   :{
                                    'phase_less_loss'       : hyp_nac2['phase_less_loss'],
                                    'reinit_weights'        : hyp_nac2['reinit_weights'],
                                    'val_disjoint'          : hyp_nac2['val_disjoint'],
                                    'val_split'             : hyp_nac2['val_split'],
                                    'epo'                   : hyp_nac2['epo'],
                                    'pre_epo'               : hyp_nac2['pre_epo'],
                                    'epomin'                : hyp_nac2['epomin'],
                                    'patience'              : hyp_nac2['patience'],
                                    'max_time'              : hyp_nac2['max_time'],
                                    'batch_size'            : hyp_nac2['batch_size'],
                                    'delta_loss'            : hyp_nac2['delta_loss'],
                                    'factor_lr'             : hyp_nac2['factor_lr'],
                                    'epostep'               : hyp_nac2['epostep'],
                                    'learning_rate_start'   : hyp_nac2['learning_rate_start'],
                                    'learning_rate_stop'    : hyp_nac2['learning_rate_stop' ],
                                    'learning_rate_step'    : hyp_nac2['learning_rate_step'],
                                    'epoch_step_reduction'  : hyp_nac2['epoch_step_reduction'],
                                    'use_linear_callback'   : False,
                                    'use_early_callback'    : False,
                                    'use_exp_callback'      : False,
                                    'use_step_callback'     : True,
       	       	                    },
       	       	      'retraining' :{
                                    'phase_less_loss'       : hyp_nac2['phase_less_loss'],
                                    'reinit_weights'        : hyp_nac2['t_reinit_weights'],
                                    'val_disjoint'          : hyp_nac2['t_val_disjoint'],
                                    'val_split'             : hyp_nac2['t_val_split'],
                                    'epo'                   : hyp_nac2['t_epo'],
                                    'pre_epo'               : hyp_nac2['t_pre_epo'],
                                    'epomin'                : hyp_nac2['t_epomin'],
                                    'patience'              : hyp_nac2['t_patience'],
                                    'max_time'              : hyp_nac2['t_max_time'],
                                    'batch_size'            : hyp_nac2['t_batch_size'],
                                    'delta_loss'            : hyp_nac2['t_delta_loss'],
                                    'factor_lr'             : hyp_nac2['t_factor_lr'],
                                    'epostep'               : hyp_nac2['t_epostep'],
                                    'learning_rate_start'   : hyp_nac2['t_learning_rate_start'],
                                    'learning_rate_stop'    : hyp_nac2['t_learning_rate_stop' ],
                                    'learning_rate_step'    : hyp_nac2['t_learning_rate_step'],
                                    'epoch_step_reduction'  : hyp_nac2['t_epoch_step_reduction'],
                                    'use_linear_callback'   : False,
                                    'use_early_callback'    : False,
                                    'use_exp_callback'      : False,
                                    'use_step_callback'     : True,            
       	       	                    },
                      'resample'   :{
                                    'phase_less_loss'       : hyp_nac2['phase_less_loss'],
                                    'reinit_weights'        : hyp_nac2['a_reinit_weights'],
                                    'val_disjoint'          : hyp_nac2['a_val_disjoint'],
                                    'val_split'             : hyp_nac2['a_val_split'],
                                    'epo'                   : hyp_nac2['a_epo'],
                                    'pre_epo'               : hyp_nac2['a_pre_epo'],
                                    'epomin'                : hyp_nac2['a_epomin'],
                                    'patience'              : hyp_nac2['a_patience'],
                                    'max_time'              : hyp_nac2['a_max_time'],
                                    'batch_size'            : hyp_nac2['a_batch_size'],
                                    'delta_loss'            : hyp_nac2['a_delta_loss'],
                                    'factor_lr'             : hyp_nac2['a_factor_lr'],
                                    'epostep'               : hyp_nac2['a_epostep'],
                                    'learning_rate_start'   : hyp_nac2['a_learning_rate_start'],
                                    'learning_rate_stop'    : hyp_nac2['a_learning_rate_stop' ],
                                    'learning_rate_step'    : hyp_nac2['a_learning_rate_step'],
                                    'epoch_step_reduction'  : hyp_nac2['a_epoch_step_reduction'],
                                    'use_linear_callback'   : False,
                                    'use_early_callback'    : False,
                                    'use_exp_callback'      : False,
                                    'use_step_callback'     : True, 	
       	       	                    },
                      }

        ## prepare training data
        self.version    = variables_all['version']
        self.ncpu       = variables_all['control']['ml_ncpu']
        self.pred_data  = variables['pred_data']
        self.train_mode = variables['train_mode']
        self.shuffle    = variables['shuffle']

        if self.train_mode not in ['training','retraining','resample']:
            self.train_mode = 'training'

        if id == None or id == 1:
            self.name   = f"NN-{title}"
        else:
            self.name   = f"NN-{title}-{id}"
        self.silent     = variables['silent']
        self.x          = data['xyz'][:,:,1:].astype(float)
        self.y_dict     ={
        'energy_gradient' :[data['energy'],data['gradient']],
        'nac'             : data['nac'],
        }

        ## combine hypers
        self.hyper ={
        'energy_gradient' : None,
        'nac'             : None,
        }

        if nn_eg_type == 1:  # same architecture with different weight
            self.hyper['energy_gradient']=hyp_dict_eg
        else:
       	    self.hyper['energy_gradient']=[hyp_dict_eg,hyp_dict_eg2]

        if nn_nac_type == 1: # same architecture with different weight
       	    self.hyper['nac']=hyp_dict_nac
       	else:
            self.hyper['nac']=hyp_dict_nac=[hyp_dict_nac,hyp_dict_nac2]

        ## initialize model
        self.model	= NeuralNetPes(self.name)

    def _heading(self):

        headline="""
%s
 *---------------------------------------------------*
 |                                                   |
 |                  Neural Networks                  |
 |                                                   |
 *---------------------------------------------------*

""" % (self.version)
 
       	return headline

    def _whatistime(self):
        return datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')

    def _howlong(self,start,end):
        walltime=end-start
        walltime='%5d days %5d hours %5d minutes %5d seconds' % (int(walltime/86400),int((walltime%86400)/3600),int(((walltime%86400)%3600)/60),int(((walltime%86400)%3600)%60))
        return walltime

    def train(self):
        ## ferr      : dict
        ##            Fitting errors, share the same keys as y_dict

        start=time.time()

        self.model.create({'energy_gradient':self.hyper['energy_gradient'],
                           'nac'            :self.hyper['nac'],
                           })

        topline='Neural Networks Start: %20s\n%s' % (self._whatistime(),self._heading())
        runinfo="""\n  &nn fitting \n"""

        if self.silent == 0:
            print(topline)
            print(runinfo)

        log=open('%s.log' % (self.name),'w')
        log.write(topline)
        log.write(runinfo)
        log.close()

        if self.train_mode == 'resample':
            out_index,out_errr,out_fiterr,out_testerr=self.model.resample(self.x,self.y_dict,gpu_dist={},proc_async=self.ncpu>=4)
        else:
            ferr=self.model.fit(self.x,self.y_dict,gpu_dist={},proc_async=self.ncpu>=4,fitmode=self.train_mode,random_shuffle=self.shuffle)

            #self.model.save()
            err_eg1=ferr['energy_gradient'][0]
            err_eg2=ferr['energy_gradient'][1]
            err_n=ferr['nac']

            H_to_eV=27.21138624598853
            H_Bohr_to_eV_A=27.21138624598853/0.52917721090380
            Bohr_to_A=1/0.52917721090380

            train_info="""
  &nn validation mean absolute error
-------------------------------------------------------
      energy       gradient       nac
        eV           eV/A         1/A
  %12.8f %12.8f %12.8f
  %12.8f %12.8f %12.8f

""" % (err_eg1[0]*H_to_eV, err_eg1[1]*H_Bohr_to_eV_A, err_n[0]*Bohr_to_A ,err_eg2[0]*H_to_eV, err_eg2[1]*H_Bohr_to_eV_A, err_n[1]*Bohr_to_A)

        end=time.time()
        walltime=self._howlong(start,end)
        endline='Neural Networks End: %20s Total: %20s\n' % (self._whatistime(),walltime)

        if self.silent == 0:
            print(train_info)
            print(endline)

        log=open('%s.log' % (self.name),'a')
        log.write(train_info)
        log.write(endline)
        log.close()

        return self

    def load(self):
        self.model.load()

        return self

    def	appendix(self,addons):
       	## fake	function does nothing

       	return self

    def evaluate(self,x):
        ## y_pred   : dict
        ## y_std    : dict
        ##            Prediction and std, share the same keys as y_dict

        if x == None:
            with open('%s' % self.pred_data,'r') as preddata:
                pred=json.load(preddata)
            pred_natom,pred_nstate,pred_xyz,pred_invr,pred_energy,pred_gradient,pred_nac,pred_ci,pred_mo=pred
            x=np.array(pred_xyz)[:,:,1:].astype(float)
            y_pred,y_std=self.model.predict(x)
            compare=1
        else:
            atoms=len(x)
            x=np.array(x)[:,1:4].reshape([1,atoms,3]).astype(float)
            y_pred,y_std=self.model.call(x)
            compare=0


        ## NN uses eV and eV/A, but MD uses Hartree and Hartree/Bohr
        H_to_eV=27.21138624598853
        H_Bohr_to_eV_A=27.21138624598853/0.52917721090380

        e_pred=y_pred['energy_gradient'][0]#/H_to_eV
        g_pred=y_pred['energy_gradient'][1]#/H_Bohr_to_eV_A
        n_pred=y_pred['nac']
        e_std=y_std['energy_gradient'][0]  #/H_to_eV 
        g_std=y_std['energy_gradient'][1]  #/H_Bohr_to_eV_A
        n_std=y_std['nac']

        if compare == 1:
            de=np.abs(np.array(pred_energy)   - e_pred)
            dg=np.abs(np.array(pred_gradient) - g_pred)
            dn=np.abs(np.array(pred_nac)      - n_pred)
            for i in range(len(x)):
                print('%5s: %8.4f %8.4f %8.4f %8.4f %8.4f' % (i+1,de[i][0],de[i][1],np.amax(dg[i][0]),np.amax(dg[i][1]),np.amax(dn[i])))

        ## Here I will need some function to print/save output
        length=len(x)
        if self.silent == 0:
            o=open('%s-e.pred.txt' % (self.name),'w')
            p=open('%s-g.pred.txt' % (self.name),'w')
            q=open('%s-n.pred.txt' % (self.name),'w')
            np.savetxt(o,np.concatenate((e_pred.reshape([length,-1]),e_std.reshape([length,-1])),axis=1))
            np.savetxt(p,np.concatenate((g_pred.reshape([length,-1]),g_std.reshape([length,-1])),axis=1))
            np.savetxt(q,np.concatenate((n_pred.reshape([length,-1]),n_std.reshape([length,-1])),axis=1))
            o.close()
            p.close()
            q.close()

        ## in MD, the prediction shape is (1,states) for energy and (1,states,atoms,3) for forces and nacs
        ## the return data should remove the batch size 1, thus take [0] of the data
        return {
                'energy'   : e_pred[0],
                'gradient' : g_pred[0],
                'nac'      : n_pred[0],
                'civec'    : None,
                'movec'    : None,
                'err_e'    : np.amax(e_std[0]),
                'err_g'    : np.amax(g_std[0]),
                'err_n'    : np.amax(n_std[0]),
       	       	}


