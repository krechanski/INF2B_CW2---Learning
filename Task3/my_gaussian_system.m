%
% Template for my_gaussian_system.m
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

%YourCode - Prepare to measure time

% Run classification
tic
epsilon = 0.01;
[Cpreds, Ms, Covs] = my_gaussian_classify(Xtrn, Ctrn, Xtst, epsilon);
toc

%YourCode - Measure the time taken, and display it.

%YourCode - Get a confusion matrix and accuracy
[cm, acc] = my_confusion(Ctst, Cpreds);


%YourCode - Save the confusion matrix as "Task3/cm.mat".
save('cm.mat','cm');

%YourCode - Save the mean vector and covariance matrix for class 26,
%           i.e. save Mu(:,26) and Cov(:,:,26) as "Task3/m26.mat" and
%           "Task3/cov26.mat", respectively.

saveMS26 = Ms(:,26);
saveCOV26 = Covs(:,:,26);

save('m26.mat', 'saveMS26');
save('cov26.mat', 'saveCOV26');

%YourCode - Display the required information - N, Nerrs, acc.
sum=0;
   for x=1:size(cm,1)
       for j=1:size(cm,1)
           if x~=j 
               sum = sum + cm(x,j);
            end
        end
    end

display = sprintf('N: %d,  Number of errors: %d,  Accuracy: %d', size(Xtst,1), sum, acc);
disp(display);





  
