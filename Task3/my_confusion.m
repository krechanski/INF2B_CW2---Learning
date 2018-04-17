function [CM, acc] = my_confusion(Ctrues, Cpreds)
% Input:
%   Ctrues : N-by-1 ground truth label vector
%   Cpreds : N-by-1 predicted label vector
% Output:
%   CM : K-by-K confusion matrix, where CM(i,j) is the number of samples whose target is the ith class that was classified as j
%   acc : accuracy (i.e. correct classification rate)

NumClasses = unique(Ctrues);
CM = zeros(size(NumClasses,1));

numRowClasses = size(Ctrues, 1);

for i=1:size(Ctrues,1)
    ConfEntry = CM(Ctrues(i,1), Cpreds(i,1));
    ConfEntry = ConfEntry +1;
    CM(Ctrues(i,1), Cpreds(i,1)) = ConfEntry;
end
total = 0;

for i=1:max(NumClasses)
    total = total + CM(i,i);
end

acc = total/numRowClasses;

end
