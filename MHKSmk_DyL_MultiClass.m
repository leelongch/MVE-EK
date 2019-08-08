function [Vec_res , t_train] = MHKSmk_DyL_MultiClass(trainSet, testSet,C1, C2, C3, inputInf)
    t_train = 0 ;    
    totalClass = size(trainSet{1} , 2); %几分类问题
    [lenTest , dim] = size(testSet{1}); %测试集的样本个数和维度
    lenTestClassOne = size(find(testSet{1}(:,dim) == 1),1); %测试集中正样本的数量
    lenTestClassTwo = size(find(testSet{1}(:,dim) == 0),1); %测试集负正样本的数量
    testLable = testSet{1}(:,dim); %测试集的类标
    testLable(testLable==0)=2;
    resultMat = zeros(lenTest , totalClass); %测试集的个数 * 2类

    %视角数
    V = inputInf.V;
    classOne = cell(V,1);
    classTwo = cell(V,1);
    testSample = cell(V,1);
    for i = 1:V
         classOneV = trainSet{i}{1}; %取训练集中的正类i
         classTwoV = trainSet{i}{2};%取训练集中的负类j
         dimV = size(testSet{i},2);
         testSetV = testSet{i}(:,1:dimV-1);
         classOne{i} = classOneV;
         classTwo{i} = classTwoV;
         testSample{i} = testSetV;
    end
            [temp t]= MHKSmk_One2One(classOne , classTwo , testSample ,C1, C2, C3,inputInf) ; 
            t_train = t_train + t;
          
            indexClassOne = find(temp == 1); %投票为正类的样本序号
            resultMat(indexClassOne , 1) = resultMat(indexClassOne , 1) + 1; 
            indexClassTwo = find(temp == -1); %投票为负类的样本序号
            resultMat(indexClassTwo , 2) = resultMat(indexClassTwo , 2) + 1; 
    [C, finalClass] = max((resultMat')) ;%C是票数，finalClass是投票产生的类标号
    finalClass1 = finalClass';

    %统计计算分类的正确情况
    forCount = finalClass1 + testLable;
    TP = length(find(forCount==2));%少数类（Positive）里面分对的个数
    TN = length(find(forCount==4));%多数类（Negative）里面分对的个数
    FN = lenTestClassOne - TP;%少数类（Positive）里面分错的个数
    FP = lenTestClassTwo - TN;%多数类（Negative）里面分错的个数
    %计算各种评价指标
    TP_rate = TP/(TP+FN);
    TN_rate = TN/(FP+TN);
    MACC = (TP_rate+TN_rate)/2;%算术平均
    GM = sqrt(TP_rate*TN_rate);%几何平均
    F1 = 2*TP/(lenTest + TP -TN);
    %AUC = (1+TP_rate-FP_rate)*50;%AUC（无效）
    % vec_res = [TP,FP,TN,FN,Acc,Acc2,GM,AUC];
    Vec_res = [TP_rate,TN_rate,MACC,GM,F1];
end

%多类问题 一对一训练
function [temp, t_train] = MHKSmk_One2One(classOne , classTwo , testSet, C1, C2,C3,inputInf)
    t_train = 0;
    lenTest = size(testSet{1} , 1) ;%测试样本的个数
    underTest = 1 : lenTest ;
    tic;
    [classOne , classTwo, testSet] = GenerateEmpiricalData(classOne , classTwo , testSet , inputInf) ;
    w = MHKSmK_DyL_Fun(classOne , classTwo ,C1, C2,C3, inputInf);
    t = toc;
    t_train = t_train + t;
    M = inputInf.M; %M个核空间
    V=inputInf.V;
    temp = class4test(w, testSet, inputInf,M,V);
end

function temp = class4test(w, testSet, inputInf,M,V)
    mWeight = 1.0/M; 
    vWeight = 1.0/V;
    %vWeight = [0.5 0.25 0.25];
    lenTest = size(testSet{1} , 1) ; %测试样本个数
    temp = zeros(lenTest , 1);
    for j = 1:V
        tempV = zeros(lenTest , 1);
        for k = 1 : M ; %M个基分类       
            dim = size(testSet{j}{k} , 2) ;%测试集的特征维度
            tempV = tempV + (testSet{j}{k}*w{j}{k}(1:dim) + w{j}{k}(dim+1))* mWeight;
        end;
        temp = temp + tempV*vWeight;
    end
    % res = sign(1/M * temp) ;
    temp = sign(temp) ;   

%     test_set = zeros(lenTest, 1);
%     for i = 1: inputInf.M
%         data = [testData{i}, ones(lenTest, 1)];
%         test_set = test_set + data*w{i};
%     end
%     temp = sign(test_set);
%     temp(find(temp == 0)) = 1;
end

function [class_one , class_two , testData] = GenerateEmpiricalData(org_one , org_two , testSet , inputInf) 
    M = inputInf.M ;
    V = inputInf.V;
    class_one = cell(V,1);
    class_two = cell(V,1);
    testData = cell(V,1);
    for j = 1:V
        class_oneV = [] ;
        class_twoV = [] ;
        testDataV = [] ;
        temp_data = [org_one{j} ; org_two{j}] ;
        size_class_two = size(org_two{j} , 1) ;
        cellLen = floor(size_class_two / M);
         for i = 1 : M
            data4Ker = [org_one{j} ; org_two{j}(1+(i-1)*cellLen : i*cellLen,:)]; 
            tempLen = size(data4Ker , 1);
            kernelPar = aveRBFPar(data4Ker, tempLen);
            kernelType = 'rbf';%linear  rbf  poly  nopaker  nopaker2
            trainData.classOne = org_one{j};
            trainData.classTwo = org_two{j}(1+(i-1)*cellLen : i*cellLen,:);
            size_class_one = size(trainData.classOne , 1) ;
            size_class_two = size(trainData.classTwo , 1) ;
            [emp_train , emp_Test, t] = kernel_mapping(trainData , testSet{j} , kernelType , kernelPar) ;
            indX1 = 1 : size_class_one ;
            indX2 = (1 + size_class_one) : (size_class_one + size_class_two) ;
            class_oneV{i} = emp_train(indX1,:) ;
            class_twoV{i}= emp_train(indX2,:) ;
            testDataV{i} = emp_Test ; 
         end  
         class_one{j} = class_oneV;
         class_two{j} = class_twoV;
         testData{j} = testDataV;
    end
end


function par=aveRBFPar(data , size)
    mat_temp = sum(data.^2,2) * ones(1,size) + ones(size,1)*sum(data.^2,2)' - 2* data*data';
    tempMean = (1/size^2) * sum(sum(mat_temp,1),2) ;
    par = sqrt(tempMean) ;
end

