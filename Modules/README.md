# Modules

The main part of this project is this folder, the mentioned modules are here, are other functions are provided in "Utils" folder.

## Analyzing Already Existed Dataset (PSTD_AnAE.py)

To use this module you must instance an object with the class, two options are provided:

- Loading Saved Data of PSim Module (*Standard Form Data*), or
- Loading Non-Standard Form Data by passing a set of rewards, actions, and the probability of reward.
  
Functions are explained below.


    Accuracy_Performance(sublist, ax = None)

Returns the accuracy of subjects of sublist during learning process, indeed for each subject a vector will be provided which indicates that up to each trial, how many TRUE action is chosen and how many FALSE. (draw it if 'ax' is passed)

    ParametersLikelihood(sublist, **kwargs)

    kwargs = {

        'T' -> Temperature Factor,
        'a_gain' -> Gain Learning Rate,
        'a_loss' -> Loss Learning Rate
    }

Calculates the likelihood of paramters in *kwargs* given the actions and rewards.

    ACSENT(sublist, n_win = 2, base = 2)

This function calculates the number of occurence of each action given prior rewards.

$P(a_{t}|r_{t-1}, r_{t-2}, ..., r_{t-n})$

In which the $a_{t}$ is the time $t$ action, and $r_{t-m}$ is reward of trial number $t-m$.

## Population Simulation (PSTD_PSim.py)

This module may be used to simulate a subject or a population of subjects in Probabilistic Selection Task.

    PSim = PS_Task_Pop_Simulator(
        
        population_number -> Number of people in population,
        prob_class -> probability of gainin reward by choosing the true option,
        exp_length -> number of trials,
        punishment -> indicates that the non-rewarded feedbacks only contains omission of reward or means a real punishing object,
        ProbabilityCalculationMethod -> the method for calculating probability of choosing the action
    
    )

Functions are explained here.

    AssignLearningRate(a)

It it possible to assign a learning rate for whole population or each subject differently.

    AssignTemperatureFunction(T)

Does the same for Temperature Function.

    Subject_Simul()

Simulate the population/subject performance in the task, with given parameter.

    BuildAndSave(output_name, output_dir)

Save the Simulated data!
