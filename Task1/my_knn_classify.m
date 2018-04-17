function [Cpreds] = my_knn_classify(Xtrn, Ctrn, Xtst, Ks)
% Input:
%   Xtrn : M-by-D training data matrix
%   Ctrn : M-by-1 label vector for Xtrn
%   Xtst : N-by-D test data matrix
%   Ks   : L-by-1 vector of the numbers of nearest neighbours in Xtrn
% Output:
%  Cpreds : N-by-L matrix of predicted labels for Xtst

%Calculating the Eucledean Distance
DistMatrix = MySqDist(Xtst, Xtrn);


%Sorting the distances
[DistSorted, idx] = sort(DistMatrix,2, 'ascend');


%Iterating through the sorted distances to classify them
for i=1: size(Ks,1)
     j = Ks(i, 1);
     kNeighbours = DistSorted(:,1:j );
     idx_values= idx(:,1:j);
     PredictedValues = mode(Ctrn(idx_values),2);
     Cpreds(:,i) = PredictedValues;
end
end
