README - Hyperparameter seletion for models used in 

Winner et al., Discovering individual-specific gait signatures from data-driven models of neuromechanical dynamics.

These 3 tables were used to select model hyperparameters, as described in the paper's Methods. 

Tables:

TrainVal: Training and validation mean squared errors, averaged across all participants for each model
- Columns: 
---- lookback = lookback parameter in the long short-term memory layer of the recurrent neural network (samples)
---- nodes = number of nodes in the long short-term memory layer of the recurrent neural network (samples)
---- train_loss = training loss for each (node-lookback combination; some models were run for multiple initialization(degress^2)
---- validation_loss = validation loss for each model (node-lookback combination; some models were run for multiple initialization(degress^2)
- Rows: one model run, defined by the lookback parameter, number of nodes, and initialization. The three best models were run for 10 initializations each. All other models were run for 1 initialization.

GaitSigAlignment_ShortTime: Short-time gait signature alignment for the 3 best models (coefficient of determination)
- Columns: Mean (mean) and standard deviation (sd) across trials for gait signature alignment (R^2)
---- Each column is listed as [mean/sd_[number of nodes]_[lookback parameter]. 
- Rows: Gait signature alignment for each initialization of the recurent neural network (10 intializations per model).

GaitSigAlignment_LongTime: Long-time gait signature alignment for the 3 best models (coefficient of determination)
- Columns: Mean (mean) and standard deviation (sd) across trials for gait signature alignment (R^2)
---- Each column is listed as [mean/sd_[number of nodes]_[lookback parameter]. 
- Rows: Gait signature alignment for each initialization of the recurent neural network (10 intializations per model).