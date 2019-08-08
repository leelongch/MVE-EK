function [Segment,Segment2,Segment3] = samples2Pieces(dataSet,dataSet2, dataSet3 , segmentNum) 
    %
    %dataSet is a structrue of {Class_1 , Class_2 , Class_3 , ...}
    %
    totalClass = size(dataSet , 2) ;
    Segment = [] ;
    Segment2 = [] ;
    Segment3 = [] ;
    for i = 1 : totalClass
        classData = dataSet{1,i} ;
        classData2 = dataSet2{1,i} ;
        classData3 = dataSet3{1,i} ;
        len = size(classData , 1) ;
        index = randperm(len) ;
        segSize = floor(len/segmentNum) ;
        for k = 1 : segmentNum - 1
            Segment{i,k} = classData(index(segSize*(k-1) + 1 : segSize*k) , :) ;
            Segment2{i,k} = classData2(index(segSize*(k-1) + 1 : segSize*k) , :) ;
            Segment3{i,k} = classData3(index(segSize*(k-1) + 1 : segSize*k) , :) ;
        end
        Segment{i,k+1} = classData(index(segSize*(k) + 1 : len) , :) ;
        Segment2{i,k+1} = classData2(index(segSize*(k) + 1 : len) , :) ;
        Segment3{i,k+1} = classData3(index(segSize*(k) + 1 : len) , :) ;
    end
    Segment = Segment' ;
    Segment2 = Segment2' ;
    Segment3 = Segment3' ;
end