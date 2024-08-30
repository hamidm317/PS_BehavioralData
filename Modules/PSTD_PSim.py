import numpy as np
import pickle
from Utils import Constants

class PS_Task_Pop_Simulator():

    ## may add imbalance feedbacks!
    ## Uploading an existing dataset!

    def __init__(self, population_number, prob_class, exp_length, punishment = False, ProbabilityCalculationMethod = 'softmax'):

        assert population_number > 4, "Population must be at least include 5 members"
        assert prob_class >= 0.5 and prob_class <= 1, "The prob_class must be between 0.5 and 1 (the reward probability of the more rewarded one)"

        self.population = population_number
        self.PrCl = prob_class
        self.EL = exp_length

        self.PunMiss = punishment
        self.ProbGenMet = ProbabilityCalculationMethod

    def AssignLearningRate(self, a):

        assert type(a) == float or type(a) == list or type(a) == np.ndarray, "Learning rate must be a float number or array of float numbers"

        if type(a) == float:

            print("Whole population have a similar learning rate")

            self.LR = a
            self.LR_HG = 0

        else:

            a = np.array(a)

            assert a.ndim < 3, "Learning rate array should have less than 3 dimensions"

            if a.ndim == 0:

                print("Whole population have a similar learning rate")

                self.LR = a[0]
                self.LR_HG = 0

            elif a.ndim == 1:

                assert a.shape[0] == self.population or a.shape[0] == 2, "Number of assigned learning rates doesn't match the population"

                if a.shape[0] == self.population:

                    print("You assign a different Learning Rate for each member")

                    self.LR = a
                    self.LR_HG = 1

                else:

                    if a.shape[0] == 2:

                        print("You assign different gain and loss Learning Rate")

                        self.LR = a
                        self.LR_HG = 2              

            elif a.ndim == 2:

                assert a.shape[0] == 2 or a.shape[1] == 2, "The assigned learning rates of each subject must be maiximum two"
                assert a.shape[0] == self.population or a.shape[1] == self.population, "Number of assigned learning rates doesn't match the population"

                if a.shape[0] == 2:

                    a = a.T

                self.LR = a
                self.LR_HG = 3

    def AssignTemperatureFactor(self, T):

        assert type(T) == float or type(T) == list or type(T) == np.ndarray, "TF must be a float number or array of float numbers"

        if type(T) == float:

            print("Whole population have a similar TF")

            self.TF = T
            self.TF_HG = 0

        else:

            T = np.array(T)

            assert T.ndim < 2, "Learning rate array should have less than 2 dimensions"

            if T.ndim == 0:

                print("Whole population have a similar TF")

                self.TF = T[0]
                self.TF_HG = 0

            elif T.ndim == 1:

                assert T.shape[0] == self.population, "Number of assigned learning rates doesn't match the population"

                print("You assign a different Learning Rate for each member")

                self.TF = T
                self.TF_HG = 1

    def Subject_Simul(self):

        # This Functions simulates a subject in the Probabilistic Learning Task with given Arguments
        # # Prob_Class -> Reward chance for the more rewarded stimulus
        # # Leangth -> The number of trials
        # # T -> Temperature factor, indicates the Explore/Exploit Tendency
        # # a -> The learning rate:
        # # # a: float number or single-index array/list -> a unique learning rate for both gain and loss
        # # # a: double-indexed array/list -> a = [a_gain, a_loss]. learning rates for gain and loss, respectively

        self.G_Choices = []
        self.G_Rewards = []
        self.G_Q = []

        G_Choices = []
        G_Rewards = []
        G_Q = []

        for mem in range(self.population):

            if self.LR_HG == 0:

                a = self.LR

            elif self.LR_HG == 1:

                a = self.LR[mem]

            elif self.LR_HG == 2:

                a = self.LR

            elif self.LR_HG == 3:

                a = self.LR[mem, :]

            else:

                print("This is a bug in the class code")

                return False

            if self.TF_HG == 0:

                T = self.TF

            elif self.TF_HG == 1:

                T = self.TF[mem]

            else:

                print("This is a bug in the class code")

                return False

            Experiment = self.Experiment_Generator()

            Q = np.array([0.0, 0.0])

            Choices = []
            Rewards = []

            for i, Trial in enumerate(Experiment):

                Choice = self.Decision_Maker(self.Choice_Probability(Q, 0, T), 0)

                Reward = self.Reward_Generator(Trial, Choice)

                if type(a) == float:

                    Q[Choice] = Q[Choice] + a * (Reward - Q[Choice])

                elif len(a) == 1:

                    Q[Choice] = Q[Choice] + a[0] * (Reward - Q[Choice])

                else:

                    Q[Choice] = Q[Choice] + a[0] * np.max([Reward - Q[Choice], 0]) + a[1] * np.min([Reward - Q[Choice], 0])

                Choices.append(Choice)
                Rewards.append(Reward)

            G_Choices.append(Choices)
            G_Rewards.append(Rewards)
            G_Q.append(Q)

        self.G_Choices = np.array(G_Choices)
        self.G_Rewards = np.array(G_Rewards)
        self.G_Q = np.array(G_Q)

    def Accuracy_Performance(self, sublist, ax = None):

        assert type(sublist) == int or type(sublist) == list or type(sublist) == np.ndarray, "Subjects list/number must be integer or list/array of integers"

        if type(sublist) == int:

            assert sublist >= 0 and sublist < self.population, "Insert valid subject number"

            Acc = []

            for i in range(self.EL):

                Acc.append(1 - np.sum(self.G_Choices[sublist, :i]) / (i + 1))

            if ax != None:

                ax.plot(np.arange(self.EL), Acc)
                ax.set_xlabel("Trial Nmuber")
                ax.set_ylabel("Accuracy")
                ax.set_title("Accuracy Performance")

            return np.array(Acc)

        else:

            Acc = []

            sublist = np.array(sublist)

            assert sublist.ndim == 1, "List of Subjects must be 1-Dimensional"
            assert np.all(sublist < self.population) and np.all(sublist >= 0), "Invalid subject in list"

            for sub in sublist:

                Acc_sub = []

                for i in range(self.EL):

                    Acc_sub.append(1 - np.sum(self.G_Choices[sub, :i]) / (i + 1))

                if ax != None:

                    ax.plot(np.arange(self.EL), Acc_sub)
                    ax.set_xlabel("Trial Nmuber")
                    ax.set_ylabel("Accuracy")
                    ax.set_title("Accuracy Performance")

                Acc.append(Acc_sub)

            if ax != None:

                ax.plot(np.arange(self.EL), np.mean(np.array(Acc), axis = 0), label = 'Mean Performance', color = 'k')
                ax.legend()

            return np.array(Acc)

    def BuildAndSave(self, output_name, output_dir = Constants.LocalDataConstants['BehSimSaveDir']):

        self.Pop_Perf = {'PopNum': self.population, 'Exp Length': self.EL, 'Prob Class': self.PrCl, 'LR': self.LR, 'TempFact': self.TF, 'Actions': self.G_Choices, 'Feedbacks': self.G_Rewards}

        with open(output_dir + output_name, 'wb') as f:
            
            pickle.dump(self.Pop_Perf, f)

    def Experiment_Generator(self): # Generate a random Experiment, as environment

    # # The 0 trials reward on the more rewarded stimulus as vice versa.

        return np.random.permutation(np.concatenate((np.zeros(int(self.PrCl * self.EL)), np.ones(int((1 - self.PrCl) * self.EL + 1)))))

    def Reward_Generator(self, Trial, Choice): # Generates the Reward of Choice in the given Trial (0 Choice reward on 0 and 1 on 1)

        if self.PunMiss:

            return 1 - np.abs(Trial - Choice)

        else:

            return 1 - 2 * np.abs(Trial - Choice)

    def Choice_Probability(self, Q, Choice, T): # Generate The Probability of Selecting a Stimulation

        if self.ProbGenMet == 'softmax':

            return np.exp(Q[Choice] / T) / (np.exp(Q[Choice] / T) + np.exp(Q[1 - Choice] / T))

        else:

            print("Only Softmax is defined ^-^")

            return False

    def Decision_Maker(self, Q_Prob, Choice):

        if np.random.random() < Q_Prob:

            return Choice
        
        else:

            return 1 - Choice