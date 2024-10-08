import h5py
from pandas import read_excel, read_csv
import numpy as np
import ast

from Utils.Constants import LocalDataConstants

def ExperimentDataLoader():

    BehavioralData = read_excel(LocalDataConstants.directories['beh_dir_file'])
    Performance_data = read_csv(LocalDataConstants.directories['perform_data_dir'])

    return BehavioralData, Performance_data

def AvailableSubjects():

    SOI = np.load(LocalDataConstants.directories['ListOfAvailableSubjects'])

    return SOI

def ActionRewardExtractor(stim, dataframe_dir = r'E:\HWs\Msc\Research\Research\Depression Dataset\New Datasets\Subjects_Behavioral_datas.csv'):
    
    if type(stim) == int:

        stim = [stim]
    
    dataframe = read_csv(dataframe_dir)

    all_stims_action_list = []
    all_stims_reward_list = []
    prob_classes = []

    if stim == 'All':

        all_stim_action_list = []
        all_stim_reward_list = []

        for sample in range(len(dataframe)):
            
            list_ = ast.literal_eval(dataframe['Task'][sample])
            Tasks_matrix = np.array(list_)

            all_stim_action_list.append(np.squeeze(Tasks_matrix[:, 2]))
            all_stim_reward_list.append((np.squeeze(Tasks_matrix[:, 3]) + 1) / 2)

        prob_class = 0.7   


        all_stims_action_list.append(all_stim_action_list)
        all_stims_reward_list.append(all_stim_reward_list)
        
        prob_classes.append(prob_class)

    else:

        for stim_ in stim:

            all_stim_action_list = []
            all_stim_reward_list = []

            for sample in range(len(dataframe)):
                
                list_ = ast.literal_eval(dataframe['Task'][sample])
                Tasks_matrix = np.array(list_)

                all_stim_action_list.append(np.squeeze(Tasks_matrix[np.where(Tasks_matrix[:, 0] == stim_), 2]))
                all_stim_reward_list.append((np.squeeze(Tasks_matrix[np.where(Tasks_matrix[:, 0] == stim_), 3]) + 1) / 2)

            prob_class = 0.8 - stim_ * 0.1    


            all_stims_action_list.append(all_stim_action_list)
            all_stims_reward_list.append(all_stim_reward_list)
            
            prob_classes.append(prob_class)

    return all_stims_action_list, all_stims_reward_list, prob_classes

def LoadCavanaghEstParams(AvailableSubjects, excel_dir = r'E:\HWs\Msc\Research\Research\Depression Dataset\depression_rl_eeg\Depression PS Task\Scripts from Manuscript\Data_4_Import.xlsx'):

    tmp_Data = read_excel(excel_dir)

    return np.array(tmp_Data['TST_aG'][AvailableSubjects]), np.array(tmp_Data['TST_aL'][AvailableSubjects])

def ReactionTimeExtractor(stim, dataframe_dir = r'E:\HWs\Msc\Research\Research\Depression Dataset\New Datasets\Subjects_Behavioral_datas.csv'):
    
    if type(stim) == int:

        stim = [stim]
    
    dataframe = read_csv(dataframe_dir)

    all_stims_rt_list = []
    prob_classes = []

    if stim == 'All':

        all_stim_rt_list = []

        for sample in range(len(dataframe)):
            
            list_ = ast.literal_eval(dataframe['Task'][sample])
            Tasks_matrix = np.array(list_)

            all_stim_rt_list.append(np.squeeze(Tasks_matrix[:, 1]))

        prob_class = 0.7   


        all_stims_rt_list.append(all_stim_rt_list)
        
        prob_classes.append(prob_class)

    else:

        for stim_ in stim:

            all_stim_rt_list = []

            for sample in range(len(dataframe)):
                
                list_ = ast.literal_eval(dataframe['Task'][sample])
                Tasks_matrix = np.array(list_)

                all_stim_rt_list.append(np.squeeze(Tasks_matrix[np.where(Tasks_matrix[:, 0] == stim_), 1]))

            prob_class = 0.8 - stim_ * 0.1    


            all_stims_rt_list.append(all_stim_rt_list)
            
            prob_classes.append(prob_class)

    return all_stims_rt_list, prob_classes