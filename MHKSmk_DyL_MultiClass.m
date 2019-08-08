function [Vec_res , t_train] = MHKSmk_DyL_MultiClass(trainSet, testSet,C1, C2, C3, inputInf)
    t_train = 0 ;    
    totalClass = size(trainSet{1} , 2); %����������
    [lenTest , dim] = size(testSet{1}); %���Լ�������������ά��
    lenTestClassOne = size(find(testSet{1}(:,dim) == 1),1); %���Լ���������������
    lenTestClassTwo = size(find(testSet{1}(:,dim) == 0),1); %���Լ���������������
    testLable = testSet{1}(:,dim); %���Լ������
    testLable(testLable==0)=2;
    resultMat = zeros(lenTest , totalClass); %���Լ��ĸ��� * 2��

    %�ӽ���
    V = inputInf.V;
    classOne = cell(V,1);
    classTwo = cell(V,1);
    testSample = cell(V,1);
    for i = 1:V
         classOneV = trainSet{i}{1}; %ȡѵ�����е�����i
         classTwoV = trainSet{i}{2};%ȡѵ�����еĸ���j
         dimV = size(testSet{i},2);
         testSetV = testSet{i}(:,1:dimV-1);
         classOne{i} = classOneV;
         classTwo{i} = classTwoV;
         testSample{i} = testSetV;
    end
            [temp t]= MHKSmk_One2One(classOne , classTwo , testSample ,C1, C2, C3,inputInf) ; 
            t_train = t_train + t;
          
            indexClassOne = find(temp == 1); %ͶƱΪ������������
            resultMat(indexClassOne , 1) = resultMat(indexClassOne , 1) + 1; 
            indexClassTwo = find(temp == -1); %ͶƱΪ������������
            resultMat(indexClassTwo , 2) = resultMat(indexClassTwo , 2) + 1; 
    [C, finalClass] = max((resultMat')) ;%C��Ʊ����finalClass��ͶƱ����������
    finalClass1 = finalClass';

    %ͳ�Ƽ���������ȷ���
    forCount = finalClass1 + testLable;
    TP = length(find(forCount==2));%�����ࣨPositive������ֶԵĸ���
    TN = length(find(forCount==4));%�����ࣨNegative������ֶԵĸ���
    FN = lenTestClassOne - TP;%�����ࣨPositive������ִ�ĸ���
    FP = lenTestClassTwo - TN;%�����ࣨNegative������ִ�ĸ���
    %�����������ָ��
    TP_rate = TP/(TP+FN);
    TN_rate = TN/(FP+TN);
    MACC = (TP_rate+TN_rate)/2;%����ƽ��
    GM = sqrt(TP_rate*TN_rate);%����ƽ��
    F1 = 2*TP/(lenTest + TP -TN);
    %AUC = (1+TP_rate-FP_rate)*50;%AUC����Ч��
    % vec_res = [TP,FP,TN,FN,Acc,Acc2,GM,AUC];
    Vec_res = [TP_rate,TN_rate,MACC,GM,F1];
end

%�������� һ��һѵ��
function [temp, t_train] = MHKSmk_One2One(classOne , classTwo , testSet, C1, C2,C3,inputInf)
    t_train = 0;
    lenTest = size(testSet{1} , 1) ;%���������ĸ���
    underTest = 1 : lenTest ;
    tic;
    [classOne , classTwo, testSet] = GenerateEmpiricalData(classOne , classTwo , testSet , inputInf) ;
    w = MHKSmK_DyL_Fun(classOne , classTwo ,C1, C2,C3, inputInf);
    t = toc;
    t_train = t_train + t;
    M = inputInf.M; %M���˿ռ�
    V=inputInf.V;
    temp = class4test(w, testSet, inputInf,M,V);
end

function temp = class4test(w, testSet, inputInf,M,V)
    mWeight = 1.0/M; 
    vWeight = 1.0/V;
    %vWeight = [0.5 0.25 0.25];
    lenTest = size(testSet{1} , 1) ; %������������
    temp = zeros(lenTest , 1);
    for j = 1:V
        tempV = zeros(lenTest , 1);
        for k = 1 : M ; %M��������       
            dim = size(testSet{j}{k} , 2) ;%���Լ�������ά��
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

