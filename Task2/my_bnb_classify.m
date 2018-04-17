function [Cpreds] = my_bnb_classify(Xtrn, Ctrn, Xtst, threshold)
% Input:
%   Xtrn : M-by-D training data matrix
%   Ctrn : M-by-1 label vector for Xtrn
%   Xtst : N-by-D test data matrix
%   threshold : A scalar parameter for binarisation
% Output:
%  Cpreds : N-by-1 matrix of predicted labels for Xtst

%YourCode - binarisation of Xtrn and Xtst.
M = size(Xtrn, 1); %total number of documents in the train set
N = size(Xtst, 1);
D = size(Xtrn, 2);
binary_Xtrn = zeros(M,D);
binary_Xtst = zeros(N,D);

NumClasses = unique(Ctrn);
classes = size(NumClasses,1);

%Iterating through the rows and columns and fill the cells with 1's if it's
%bigger than the threshold

for i=1:M
    for j=1:D
        if Xtrn(i,j) >= threshold 
            binary_Xtrn(i,j) = 1;
        end
    end
end

%Same thing as above is done here but this time for the Xtst matrix
for i=1:N
    for j=1:D
        if Xtst(i,j) >= threshold 
            binary_Xtst(i,j) = 1;
        end
    end
end

%YourCode - naive Bayes classification with multivariate Bernoulli distributions

%Iterate through the classes to find out how many times it occurs in a
%document
M_class = zeros(classes,1);
for class=1:classes %2
    for i=1:M
        if Ctrn(i) == class
            M_class(class) = M_class(class) + 1;
        end
    end
end

rowCount = (size(Xtrn,2));
doc_count = zeros(rowCount,classes); 

%Iterate through the classes,feature vectors and document numbers and
%sum the document count if a certain document is equal to some class
for class=1:classes 
    for feature=1:D
        for doc_num=1:M
            if Ctrn(doc_num) == class
                doc_count(feature, class) = doc_count(feature, class) + binary_Xtrn(doc_num, feature);
            end
        end
    end
end

%Prealocate space to save some time
probability_matrix = zeros(rowCount,classes); 
   
%Iterate through each class and fill the probability matrix 
for class=1:classes %2
    for feature=1:D
       probability_matrix(feature, class) = doc_count(feature, class)/M_class(class);
    end
end

prior_probability = repmat(1/classes, 1, classes); %2

post_probability = zeros(classes, N); %2

%Iterate through each document and class and compute the posteriror
%probability for each feature vector and finally return the post
%probability of the class and document number
for doc_num=1:N
    for class=1:classes %2
        product = 1;
        for feature=1:D
            b = binary_Xtst(doc_num, feature);
            P = probability_matrix(feature, class);
            x = (b * P + (1-b) * (1-P));
            if x==0
                x = 0.00001;
            end
            product = product * x;
        end
        post_probability(class, doc_num) = prior_probability(class) * product;
    end
end

    
[~,Cpreds] = max(post_probability.', [], 2);

end
