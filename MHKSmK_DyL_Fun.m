function [w1] = MHKSmK_DyL_Fun(classOne , classTwo ,C1, C2,C3, inputInf) 

    M = inputInf.M; %M¸öºË¿Õ¼ä
    V = inputInf.V;
    for j = 1: inputInf.V
    for i = 1: inputInf.M
        len_one = size(classOne{j}{i}, 1);
        len_two = size(classTwo{j}{i}, 1);
        len_all = len_one + len_two;
        one{j}{i} = ones(len_all, 1);
        %IR = inputInf.unbalanceR*len_two/len_one;
        IR1 = len_two/len_all;
        IR2 = len_one/len_all;
        D{j}{i} = diag([IR1*ones(1, len_one), IR2*ones(1, len_two)]);
        Y{j}{i} = [[classOne{j}{i}, ones(size(classOne{j}{i}, 1), 1)]; -1*[classTwo{j}{i}, ones(size(classTwo{j}{i}, 1), 1)]];
        dim = size(Y{j}{i}, 2);
        w0{j}{i} = 0.5*ones(dim, 1);
        b0{j}{i} = ones(len_all, 1)*inputInf.B(i);
        I{j}{i} = eye(dim);
        I{j}{i}(end, end) = 0;
        %P{j}{i} = pinv((1 + C2*(inputInf.M - 1)/inputInf.M)*Y{j}{i}'*D{j}{i}*Y{j}{i} + C1*I);
        feaC = size(Y{j}{i},2);
        XOne = Y{j}{i}(find(Y{j}{i}(:,feaC)==1),:); 
        P{j}{i} = pinv((C2*((M-1)/M)^2 * XOne'* XOne +C3*((V-1)/V)^2 * Y{j}{i}'*Y{j}{i}) + Y{j}{i}'*D{j}{i}*Y{j}{i}+C1*I{j}{i});
    end
    end
    [L0, mean_out, b1] = getL(w0, b0, Y, one, D, inputInf, C1, C2,C3);
    b0 = b1;
    iter = 1;
    while iter <= inputInf.sizeIter
        iter = iter + 1;
        for j = 1: inputInf.V
        for i = 1:inputInf.M
           %w1{i} = P{i}*Y{i}'*D*(b0{i} + one_all + C2*(mean_out - Y{i}*w0{i})/inputInf.M);  
           feaC = size(Y{j}{i},2);
           XOne = Y{j}{i}(find(Y{j}{i}(:,feaC)==1),:); 
           w1{j}{i} =  P{j}{i} *(Y{j}{i}'*(D{j}{i}*one{j}{i}+D{j}{i}*b0{j}{i} + C3*((V-1)/V^2)*sumWV(w0,i,j,V,Y))  +  C2*((M-1)/M^2)*XOne'*sumW(w0{j},1,M,Y{j},1));
    
        end
        end
        w0 = w1;
        [L1, mean_out, b1] = getL(w0, b0, Y, one, D, inputInf, C1, C2,C3);
        if (L1 - L0)'*(L1 - L0) <= inputInf.termination 
            break;
        end
        L0 = L1;
        b0 = b1;
    end
end

function [value, mean_out, b1] = getL(w, B, Y, one, D, inputInf, C1, C2,C3)
   lastValue1 = 0;
   lastValue2 = 0;
   mean_out = 0;
    for j = 1:inputInf.V
        tempValue1 = 0 ;
        tempValue2 = 0 ;
        tempValue3 = 0 ;
        for i = 1 : inputInf.M
            tempValue1=tempValue1+(Y{j}{i}*w{j}{i}-one{j}{i}-B{j}{i})'*D{j}{i}*(Y{j}{i}*w{j}{i}-one{j}{i}-B{j}{i}) + C1*(w{j}{i}(1:end-1))'*(w{j}{i}(1:end-1));   
            mean_out = mean_out + Y{j}{i}*w{j}{i};
            e{i} = Y{j}{i}*w{j}{i} - B{j}{i} - one{j}{i};
            b1{j}{i} = B{j}{i} + 0.99*(e{i} + abs(e{i}));
            feaC = size(Y{j}{i},2);
            XOne = Y{j}{i}(find(Y{j}{i}(:,feaC)==1),:);
            tempValue2=tempValue2+(XOne*w{j}{i}-1/inputInf.M*sumW(w{j},0,inputInf.M,Y{j},1))'*(XOne*w{j}{i}-1/inputInf.M*sumW(w{j},0,inputInf.M,Y{j},1));
            tempValue3 = tempValue3 + (Y{j}{i}*w{j}{i}-1/inputInf.V*sumWV(w ,i ,0 ,inputInf.V, Y))'*(Y{j}{i}*w{j}{i}-1/inputInf.V*sumWV(w ,i ,0 ,inputInf.V, Y));
        end;
        %lambÊÇ C(M+1)
        lastValue1 = lastValue1 +tempValue1 + C2*tempValue2 ;
        lastValue2 = lastValue2 + tempValue3;
    end
    mean_out = mean_out/(inputInf.M*inputInf.V);
    %gama C(M+2)
    value = lastValue1 + C3*lastValue2;
end

function sumXW = sumW(w,i,M,X,onlyOne)
    if i == 1 
        if onlyOne == 1
            feaC = size(X{1},2);
            sumXW = zeros(size(X{1}(find(X{1}(:,feaC)==1),:),1),1);
            for k = 2 : M ;
                feaC = size(X{k},2);
                XOne = X{k}(find(X{k}(:,feaC)==1),:);
                sumXW = sumXW + XOne*w{k};
            end
        else
            sumXW = zeros(size(X{1},1),1);
            for k = 2 : M ;
                sumXW = sumXW + X{k}*w{k};
            end;

        end
    end;

    if i == M
        if onlyOne == 1
            feaC = size(X{1},2);
            sumXW = zeros(size(X{1}(find(X{1}(:,feaC)==1),:),1),1);
            for k = 1 : M-1;
                feaC = size(X{k},2);
                XOne = X{k}(find(X{k}(:,feaC)==1),:);
                sumXW = sumXW + XOne*w{k};
            end;
        else
            sumXW = zeros(size(X{1},1),1);
            for k = 1 : M-1;
                sumXW = sumXW + X{k}*w{k};
            end;
        end
    end;

    if i > 1 && i < M
        
        if onlyOne == 1
            feaC = size(X{1},2);
            sumXW = zeros(size(X{1}(find(X{1}(:,feaC)==1),:),1),1);
            for k = 1 : i-1;
                feaC = size(X{k},2);
                XOne = X{k}(find(X{k}(:,feaC)==1),:);
                sumXW = sumXW + Xone*w{k} ;
            end;
            for k = i+1 : M;
                feaC = size(X{k},2);
                XOne = X{k}(find(X{k}(:,feaC)==1),:);
                sumXW = sumXW + XOne*w{k};
            end;  
        else
            sumXW = zeros(size(X{1},1),1);
            for k = 1 : i-1;
                sumXW = sumXW + X{k}*w{k};
            end;
            for k = i+1 : M;
                sumXW = sumXW + X{k}*w{k};
            end;  
        end
    end;
    if i == 0 
        if onlyOne == 1
             feaC = size(X{1},2);
             sumXW = zeros(size(X{1}(find(X{1}(:,feaC)==1),:),1),1);
             for k = 1 : M
                  feaC = size(X{k},2);
                  XOne = X{k}(find(X{k}(:,feaC)==1),:);
                  sumXW = sumXW + XOne*w{k};
             end
        else
            sumXW = zeros(size(X{1},1),1);
            for k = 1 : M
                sumXW = sumXW + X{k}*w{k};
            end
         end
    end;
end

function sumXW = sumWV(w,i,j,V,X)

    sumXW = zeros(size(X{1}{1},1),1) ;
    if j == 1 ;
        for k = 2 : V ;
            sumXW = sumXW + X{k}{i}*w{k}{i} ;
        end;
    end;

    if j == V ;
        for k = 1 : V-1;
            sumXW = sumXW + X{k}{i}*w{k}{i};
        end;
    end;

    if j > 1 && j < V;
        for k = 1 : j-1;
            sumXW = sumXW + X{k}{i}*w{k}{i} ;
        end;
        for k = j+1 : V;
            sumXW = sumXW + X{k}{i}*w{k}{i};
        end;    
    end;

    if j == 0 ;
        for k = 1 : V ;
            sumXW = sumXW + X{k}{i}*w{k}{i};
        end
    end
end