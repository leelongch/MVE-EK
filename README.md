# MVE-EK
Multi-view Ensemble Learning with Empirical Kernel for Heart Failure Mortality Prediction

The code is developed under Matlab 2015b

For the specific use of each function, please see the comments for the function.

# Demo Code

[Vec_res t_train] =MHKSmk_DyL_MultiClass(trainSet, testSet,C1, C2,C3, inputInf); 

TPR = vec_res(1);

TNR = vec_res(2);

Acc = vec_res(3); AA = vec_res(4);

GM = vec_res(5);

F1 = vec_res(6);

AUC = vec_res(7);
