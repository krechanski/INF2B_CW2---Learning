%
% Template for my_bnb_system.m
%
% load the data set
%   NB: replace <UUN> with your actual UUN.
load('/afs/inf.ed.ac.uk/group/teaching/inf2b/cwk2/d/s1527764/data.mat');

% Feature vectors: Convert uint8 data to double (but do not divide by 255)
Xtrn = double(dataset.train.images);
Xtst = double(dataset.test.images);
% Labels
Ctrn = dataset.train.labels;
Ctst = dataset.test.labels;

%YourCode - Prepare to measure time

% Run classification
threshold = 1;
tic
Cpreds = my_bnb_classify(Xtrn, Ctrn, Xtst, threshold);
toc

%YourCode - Measure the time taken, and display it.

%YourCode - Get a confusion matrix and accuracy
[cm, acc] = my_confusion(Ctst, Cpreds);

%YourCode - Save the confusion matrix as "Task2/cm.mat".
save('cm.mat', 'cm');

%YourCode - Display the required information - N, Nerrs, acc.
numClass = max(Ctrn);

sum=0;
   for x=1:numClass
       for j=1:numClass
           if x~=j 
               sum = sum + cm(x,j);
            end
        end
    end

display = sprintf('N: %d,  Number of errors: %d,  Accuracy: %d', size(Xtst,1), sum, acc);
disp(display);




  
