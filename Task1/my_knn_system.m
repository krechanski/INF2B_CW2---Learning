%
% Template for my_knn_system.m
%
% load the data set
%   NB: replace <UUN> with your actual UUN.
load('/afs/inf.ed.ac.uk/group/teaching/inf2b/cwk2/d/s1527764/data.mat');

% Feature vectors: Convert uint8 data to double, and divide by 255.
Xtrn = double(dataset.train.images) ./ 255.0;
Xtst = double(dataset.test.images) ./ 255.0;

% Labels
Ctrn = dataset.train.labels;
Ctst = dataset.test.labels;

%YourCode - Prepare measuring time

% Run K-NN classification

tic
kb = [1,3,5,10,20];
Cpreds = my_knn_classify(Xtrn, Ctrn, Xtst, kb');
toc

%YourCode - Measure the time taken, and display it.

%YourCode - Get confusion matrix and accuracy for each k in kb.
%YourCode - Save each confusion matrix.

for i=1:length(kb)
    idx = kb(i);
    [cm, acc] = my_confusion(Ctst, Cpreds(:,i));
    save(sprintf('cm%d',idx), 'cm');
    
    %Sum of wrongly classified samples
    sum = 0;
    
    %Number of wrongly classified samples
    for j=1:26
        for k=1:26
            if j~=k 
                sum = sum + cm(j,k);
            end
        end
    end
    
    
    NumNeighbours = sprintf('Number of K-nearest neighbours: %d, Number of test samples: %d, Number of wrongly classified test samples: %d, Accuracy: %d',idx,size(Xtst,1), sum, acc);
                          
    disp(NumNeighbours);
end


%YourCode - Display the required information - k, N, Nerrs, acc for
%           each element of kb.








  
