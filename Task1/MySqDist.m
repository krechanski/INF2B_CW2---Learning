function [DistMatrix] = MySqDist(Mat1, Mat2)

%Get size from Xtrn and Xtst matrices
M = size(Mat2, 1);
N = size(Mat1, 1);

%Calculate the dot product
XX = dot(Mat1, Mat1, 2);
YY = dot(Mat2, Mat2, 2);

%Get the eucledean between each training point and test point using the dot
%product calculate beforehand

DistMatrix = repmat(XX,1,M)- (2*Mat1*Mat2.') + (repmat(YY,1,N)).';

end
