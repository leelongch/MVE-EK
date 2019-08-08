clc ;
load .\DataSet\xsdata_v1.mat
load .\DataSet\xsdata_v1.mat
load .\DataSet\xsdata_v1.mat
% load .\DataSet\monthxsdata_v1.mat
% load .\DataSet\monthxsdata_v2.mat
% load .\DataSet\monthxsdata_v3.mat
% load .\DataSet\yearxsdata_v1.mat
% load .\DataSet\yearxsdata_v2.mat
% load .\DataSet\yearxsdata_v3.mat

saveMatName = '.\report\xsdata_v1MV_MK_MHKS.mat';

% warning off ;
totalCycle = 5; %5轮
[train1,test1]=divide(xsdata_v1,index1,index2,totalCycle);
[train2,test2]=divide(xsdata_v2,index1,index2,totalCycle);
[train3,test3]=divide(xsdata_v3,index1,index2,totalCycle);
% [train1,test1]=divide(yearxsdata_v1,index1,index2,totalCycle);
% [train2,test2]=divide(yearxsdata_v2,index1,index2,totalCycle);
% [train3,test3]=divide(yearxsdata_v3,index1,index2,totalCycle);

%视角数
inputInf.V = 1;
V = inputInf.V;
train = cell(inputInf.V,1);
test = cell(inputInf.V,1);
train{1} = train1;
train{2} = train2;
train{3} = train3;
test{1} = test1;
test{2} = test2;
test{3} = test3;

totalClass = size(train{1}, 2); %几类问题
dim = size(train{1}{1}, 2); %特征维度
finalRecord = [] ;

inputInf.filteTime = 3;
inputInf.sizeIter = 100;
inputInf.termination = 1e-3 ;
inputInf.k = 7;
inputInf.ker = 2;

FinalRes = [] ;
CM = [1,2,3,5];
CC1 = [0.001;0.01;0.1;1;10;100];
CC2 =[0.001;0.01;0.1;1;10;100];
CC3 =[0.001;0.01;0.1;1;10;100];

for i_m = 1:length(CM)
for i_c1 = 1:length(CC1)
for i_c2 = 1:length(CC2)
for i_c3 = 1:length(CC3)
    inputInf.M = CM(i_m);
    inputInf.R = 0.99 * ones(inputInf.M,1) ;
    inputInf.B = 1e-6 * ones(inputInf.M,1) ;
    
    C1 = CC1(i_c1); 
    C2 = CC2(i_c2); %2^(-0) ;    %lamda 0.007
    C3 = CC3(i_c3); %2^(-0) ;    % gama

    fprintf('c:%f,lamda:%f,gama:%f,M:%f\n',C1, C2, C3,inputInf.M);
    res = [] ;
    resMean = [] ;
    for i_cv = 1:totalCycle; 
        trainSet = cell(inputInf.V,1);
        testSet = cell(inputInf.V,1);
        for i = 1:V
             trainSetV = cell(1 , totalClass) ;
             classoneV=train{i}{i_cv,1};
             classtwoV=train{i}{i_cv,2};
             trainSetV{1}=classoneV;
             trainSetV{2}=classtwoV;
             trainSet{i} = trainSetV;
             testSetV = test{i}{i_cv};
             testSet{i} = testSetV;
        end
        [Vec_res t_train] =MHKSmk_DyL_MultiClass(trainSet, testSet,C1, C2,C3, inputInf); 
        res = [res ; [Vec_res t_train]];
        fprintf('The %d cycle---:TP_rate,TN_rate,MACC,GM,F1,t_train\n %f,%f,%f,%f,%f,%f\n', i_cv, res(i_cv, : ));
    end;

    resMean(1 ,1 :6) = mean(res, 1); %增加一行记录每个指标的均值
    resMean(1 , 7:12) = std(res(1:totalCycle , :)) ; %增加一行记录每个指标的标准差
    resMean(1, 13:16) = [C1, C2, C3,inputInf.M] ;   %cl  lamda  gama ker
 
    %savedObj.FinalRes = res;
    FinalRes =  [FinalRes ; resMean];
    save(saveMatName , 'FinalRes');
    fprintf('TP_rate,TN_rate,MACC,GM,F1,t_train\nmean = %f,%f,%f,%f,%f,%f\n' , resMean(1,1:6));
end
end
end
end

