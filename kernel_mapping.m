function [emp_train , emp_Test , t_train] = kernel_mapping(train_data , test_data , kernelPerType , kernelPar)
    %
    % train_data = {train_class_one , train_class_two } ;
    % test_data = [X_in_input_space ] ;
    %
    
    train_class_one = train_data.classOne ;
    train_class_two = train_data.classTwo ;
    
    % len_ClassOne = size(train_class_one , 1) ;
    % len_ClassTwo = size(train_class_two , 1) ;
    
    %
    % 在特征空间中通过基提取生成 trainData 和 testData
    %
    
    [emp_train , emp_Test , t_train] = emp_Generator([train_class_one ; train_class_two] , test_data , kernelPerType , kernelPar) ;
end 

function [emp_train , emp_Test , t_train] = emp_Generator(trainData , testData , kType , kPar)
    % start clock for trainData
    tic
    
    implicitKernel = Kernel(trainData , trainData , kType , kPar) ;
    [pc , variances, explained] = pcacov(implicitKernel);

    i = 1 ;
    label = 0 ;
    while variances(i) >= 1e-3 ;
        if i+1 > size(variances,1) ;
            label = 1 ;
            break ;
        end;
        i = i + 1 ;    
    end;

    if label == 0 ;
        i = i - 1 ;
    end;

    index = [1 : i] ;
    P = pc(: , index) ;
    R = diag(variances(index)) ;
    emp_train = implicitKernel*P * R^(-1/2) ;    
    t_train = toc ;
    
    kerTestMat = Kernel(testData,trainData , kType , kPar) ;
    emp_temp = kerTestMat * P * R^(-1/2) ;  
    emp_Test = emp_temp;
end

