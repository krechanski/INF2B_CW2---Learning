load data.mat;
Xtrn = double(dataset.train.images);
Xtst = double(dataset.test.images);
Ctrn = dataset.train.labels;
Ctst = dataset.test.labels;
threshold = 30;

my_bnb_classify(Xtrn, Ctrn, Xtst, threshold);
