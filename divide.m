function [trainSet,testSet]=divide(data,index1,index2,tolcy)

      [row,col]=size(data);
        
%       for i=1:col-1
%           m=mean(ddata(:,i));
%           s=std(ddata(:,i));
%           if s==0
%               data(:,i)=ddata(:,i);
%           else
%               data(:,i)=(ddata(:,i)-m)./s;
%           end
%       end
%         
%         data(:,col)=ddata(:,col);

       segment=[];
        len1 = length(index1);
        len2=length(index2);
        segSize1 = floor(len1/tolcy) ;
        segSize2 = floor(len2/tolcy) ;
        dim=size(data,2);
        for k = 1 : tolcy - 1
            segment{k,1} = data(index1(segSize1*(k-1) + 1 : segSize1*k) , :) ;
            segment{k,2} = data(index2(segSize2*(k-1) + 1 : segSize2*k) , :) ;
        end
            segment{k+1,1} = data(index1(segSize1*(k) + 1 : len1) , :) ;
            segment{k+1,2} = data(index2(segSize2*(k) + 1 : len2) , :) ;
       trainSet = cell(tolcy ,2);
       testSet=cell(tolcy,1);
       for index_cy = 1:tolcy
            ttest = [];
             for i = 1 : 2 
                 ttest= [ttest;segment{index_cy , i}] ;
             end
             testSet{index_cy} =ttest;
            for i=1:tolcy
                if i~=index_cy
                    for j=1:2
                    trainSet{index_cy,j}=[trainSet{index_cy,j};segment{i,j}(:,1:dim-1)];%%%%%
                    end
                end
            end
       end