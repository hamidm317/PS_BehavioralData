import numpy as np
import pickle

from Utils import ProbabilityTools as PT
from Utils import VerUtils as VU

class PS_Task_Dataset():

    def __init__(self, StandardForm_Data_dir = None):

        if StandardForm_Data_dir != None:

            with open(StandardForm_Data_dir, 'rb') as f:
            
                self.Pop_Perf = pickle.load(f)

                # self.Pop_Perf = {'PopNum': self.population, 'Exp Length': self.EL, 'Prob Class': self.PrCl, 'LR': self.LR, 'TempFact': self.TF, 'Actions': self.G_Choices, 'Feedbacks': self.G_Rewards}

                self.population = self.Pop_Perf['PopNum']
                self.EL = self.Pop_Perf['Exp Length']
                self.PrCl = self.Pop_Perf['Prob Class']
                self.LR = self.Pop_Perf['LR']
                self.TF = self.Pop_Perf['TempFact']
                self.G_Choices = self.Pop_Perf['Actions']
                self.G_Rewards = self.Pop_Perf['Feedbacks']

                self.SFP = True

        else:

            self.SFP = False


    def Load_NonStandard_Behavioral_Data(self, action_list, reward_list, prob_class):

        assert self.SFP == False, "You have loaded the standard form data"
        assert len(action_list) == len(reward_list), "Inputs doesn't match"

        sub_trial_number = []

        for i in range(len(action_list)):

            actions = action_list[i]
            rewards = reward_list[i]

            assert len(actions) == len(rewards), "Subject " + str(i) + " has unmatch action and reward arrays"

            sub_trial_number.append(len(actions))

        self.G_Choices = action_list
        self.G_Rewards = reward_list
        self.EL = sub_trial_number
        self.population = len(action_list)
        self.PrCl = prob_class

    def Accuracy_Performance(self, sublist, ax = None):

        assert type(sublist) == int or type(sublist) == list or type(sublist) == np.ndarray, "Subjects list/number must be integer or list/array of integers"

        if type(sublist) == int:

            assert sublist >= 0 and sublist < self.population, "Insert valid subject number"

            sublist = [sublist]

        Acc = []

        sublist = np.array(sublist)

        assert sublist.ndim == 1, "List of Subjects must be 1-Dimensional"
        assert np.all(sublist < self.population) and np.all(sublist >= 0), "Invalid subject in list"

        if ax != None:

            print("We plot what you want, but trial lengths may be different!")
            ax.set_xlabel("Trial Number")
            ax.set_ylabel("Accuracy")

        for sub in sublist:

            Acc_sub = []

            for i in range(self.EL[sub]):

                Acc_sub.append(1 - np.sum(self.G_Choices[sub][:i]) / (i + 1))

            if ax != None:

                ax.plot(Acc_sub)

            Acc.append(Acc_sub)

        return Acc

    def StagedAccuracy(self, sublist, NumStages = 5):

        assert type(sublist) == int or type(sublist) == list or type(sublist) == np.ndarray, "Subjects list/number must be integer or list/array of integers"

        if type(sublist) == int:

            sublist = [sublist]

        StagedAcc = [1 - np.mean(self.G_Choices[sub][VU.DeterminedBlockSampling(len(self.G_Choices[sub]), NumBlock = NumStages, NumSample_inBlock = int(len(self.G_Choices[sub]) / NumStages))], axis = -1) for sub in sublist]

        return StagedAcc

    def FeedbackExperience(self, sublist, ax = None):

        assert type(sublist) == int or type(sublist) == list or type(sublist) == np.ndarray, "Subjects list/number must be integer or list/array of integers"

        if type(sublist) == int:

            assert sublist >= 0 and sublist < self.population, "Insert valid subject number"
            sublist = [sublist]

            # FeEx = []

            # for i in range(self.EL[sublist]):

            #     FeEx.append(1 - np.sum(self.G_Rewards[sublist][:i]) / (i + 1))

            # if ax != None:

            #     ax.plot(FeEx)
            #     ax.set_xlabel("Trial Number")
            #     ax.set_ylabel("Accuracy")
            #     ax.set_title("Subject " + str(sublist) + " perfromance during trials")

        # else:

        FeEx = []

        sublist = np.array(sublist)

        assert sublist.ndim == 1, "List of Subjects must be 1-Dimensional"
        assert np.all(sublist < self.population) and np.all(sublist >= 0), "Invalid subject in list"

        if ax != None:

            print("We plot what you want, but trial lengths may be different!")
            ax.set_xlabel("Trial Number")
            ax.set_ylabel("Accuracy")

        for sub in sublist:

            FeEx_sub = []

            for i in range(self.EL[sub]):

                FeEx_sub.append(1 - np.sum(self.G_Rewards[sub][:i]) / (i + 1))

            if ax != None:

                ax.plot(FeEx_sub)

            FeEx.append(FeEx_sub)

        return FeEx

    def StagedFeedbackExperience(self, sublist, NumStages = 5):

        assert type(sublist) == int or type(sublist) == list or type(sublist) == np.ndarray, "Subjects list/number must be integer or list/array of integers"

        if type(sublist) == int:

            sublist = [sublist]

        StagedFeEx = [1 - np.mean(self.G_Rewards[sub][VU.DeterminedBlockSampling(len(self.G_Rewards[sub]), NumBlock = NumStages, NumSample_inBlock = int(len(self.G_Rewards[sub]) / NumStages))], axis = -1) for sub in sublist]

        return StagedFeEx

    def ParametersLikelihood(self, sublist, **kwargs):

        options = {

            'T': [0.2],
            'a_gain': [0.1],
            'a_loss': [0.1]
        }

        assert len(kwargs) > 0, "Insert at least one and at most three parameter"
        assert type(sublist) == list or type(sublist) == np.ndarray, "Enter subjects as a list"

        if type(sublist) == list:

            sublist = np.array(sublist)

        assert np.all(sublist >= 0) and np.all(sublist < self.population), "The subjects list includes invalid IDs"

        options.update(kwargs)

        assert (type(options['T']) == list or type(options['T']) == np.ndarray) and (len(options['T']) == 1 or len(options['T']) == len(sublist)), "Insert a list/array of float number(s) for Temperature Factor, with length 1 or equal to population"
        assert (type(options['a_gain']) == list or type(options['a_gain']) == np.ndarray) and (len(options['a_gain']) == 1 or len(options['a_gain']) == len(sublist)), "Insert a list/array of float number(s) for Learning Rate, with length 1 or equal to population"
        assert (type(options['a_loss']) == list or type(options['a_loss']) == np.ndarray) and (len(options['a_loss']) == 1 or len(options['a_loss']) == len(sublist)), "Insert a list/array of float number(s) for Learning Rate, with length 1 or equal to population"

        testing_parameters = [options['T'], options['a_gain'], options['a_loss']]

        assert len(testing_parameters[1]) == len(testing_parameters[2]), "No TAXATION without REPRESENTATION"

        HG = [len(testing_parameters[i]) for i in range(2)]

        LL_values = []

        if HG[0]:

            if HG[1]:

                # Same parameters for all subjects

                for i, sub in enumerate(sublist):

                    LL_values.append(PT.Likelihood_C(self.G_Choices[sub], self.G_Rewards[sub], T_est = testing_parameters[0][0], ag_est = testing_parameters[1][0], al_est = testing_parameters[2][0]))

            else:

                # Same Temperature but different Learning Rates

                for i, sub in enumerate(sublist):

                    LL_values.append(PT.Likelihood_C(self.G_Choices[sub], self.G_Rewards[sub], T_est = testing_parameters[0][0], ag_est = testing_parameters[1][i], al_est = testing_parameters[2][i]))


        else:

            if HG[1]:

                # Same Learning Rate and different Temperature

                for i, sub in enumerate(sublist):

                    LL_values.append(PT.Likelihood_C(self.G_Choices[sub], self.G_Rewards[sub], T_est = testing_parameters[0][i], ag_est = testing_parameters[1][0], al_est = testing_parameters[2][0]))


            else:

                # Different Parameters for all subjects

                for i, sub in enumerate(sublist):

                    LL_values.append(PT.Likelihood_C(self.G_Choices[sub], self.G_Rewards[sub], T_est = testing_parameters[0][i], ag_est = testing_parameters[1][i], al_est = testing_parameters[2][i]))

        return LL_values
    
    def ACSENT(self, sublist, n_win = 2, base = 2):

        # ACtion SENsitivity To feedback

        PsA = []

        for i in sublist:

            actions = self.G_Choices[i]
            rewards = self.G_Rewards[i]

            conditions = VU.roll_mat_gen(rewards, n_win)
            nondition = actions[n_win:]

            CodedConds = PT.CodCon(conditions, base = base)
            CodedReali = PT.CodCon(np.concatenate([np.reshape(nondition, (len(nondition), 1)), conditions], axis = 1), base = base)

            Ps, _ = np.histogram(np.int32(CodedReali), bins = np.arange(2 ** (n_win + 1) + 1))

            PsA.append(Ps.reshape((2, int(len(Ps) / 2))))

        self.PsA = PsA