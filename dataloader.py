import os
import pandas as pd
import numpy as np
import pylab as plt

class DataWrapper(object):
    def __init__(self, fileInfoList, tables):
        self.rootPath = os.path.dirname(os.path.dirname(__file__))
        self.trials = pd.DataFrame(columns=['TrialID', 'ParticipantNumber', 'AFOParameter', 'WalkingSpeed'])
        kTables = []
        for i, fi in enumerate(fileInfoList):
            k = tables[i]
            k['TrialID'] = i
            kTables.append(k)
            self.trials.loc[i] = [i, fi[1], fi[2], fi[3]]
        self.kinematics = pd.concat(kTables)
        self.kinematics = self.kinematics.reset_index(drop='True')


def load_P00X(X): # For 2nd pilot dataset
    """
    Detrend and compute the phase estimate using Phaser
    INPUT:
      X -- int -- subjectID
    OUTPUT:
      k -- dataframe
    """
    rootPath = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    fileInfoList = [(os.path.join('P'+'%03d'%(X),'JointKinematics_EMG_AFO_inputs','P'+'%03d'%(X)+'_Treadmill_K0_jKINEMGAFO_inputs.csv'), 0, 'K0', 'Test') \
                    ,(os.path.join('P'+'%03d'%(X),'JointKinematics_EMG_AFO_inputs','P'+'%03d'%(X)+'_Treadmill_K1_jKINEMGAFO_inputs.csv'), 1, 'K1', 'Test') \
                    ,(os.path.join('P'+'%03d'%(X),'JointKinematics_EMG_AFO_inputs','P'+'%03d'%(X)+'_Treadmill_K3_jKINEMGAFO_inputs.csv'), 0, 'K3', 'Test') \
                    ,(os.path.join('P'+'%03d'%(X),'JointKinematics_EMG_AFO_inputs','P'+'%03d'%(X)+'_Treadmill_K5_jKINEMGAFO_inputs.csv'), 1, 'K5', 'Test')]

    tables = []
    for fi in fileInfoList:
        k = pd.read_csv(os.path.join(rootPath,fi[0]), header=0)
        tables.append(k)
    dw = DataWrapper(fileInfoList, tables)
    return dw
