# 2016ByteCup-CSCI567_Proj #
> - This repository is created by CSCI567 project which also is solution for 2016 Byte Cup.

## Branch for Scott Jiang ##
> - First try with logistic regression using xgboost.
> - Second try using collabrative filtering + xgboost.

### Cross-Validation ###
> - Using 10-fold CV to find maximum iteration of xgboost.
> - Using 10-fold CV to find optimal learning rate.
> - Set eary stopping round = 10.
> - Maximum iteration should be larger than 2000.
> - Learning rate should between 0.001 and 0.002.

### Parameters setting ###
> - num_iteration = 2000.
> - max_depth = 7.
> - eta = 0.015.

### Result ###
> - Score = 0.498189387169969.
> - Global ranking 50th/434.
> - Class ranking 1st place.

### Notes ###
> - We still have great potential to enhance the score just using CF and xgboost.

### Next Phase ###
> - Using more complex evaluation function and more models.
