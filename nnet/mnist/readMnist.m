function [train_images, train_labels, test_images, test_labels] = readMnist(dir)
if nargin == 0
    dir = mfilename('fullpath');
    dir = dir(1:length(dir)-9);
end

train_images = loadMNISTImages(strcat(dir, '/train-images-idx3-ubyte'));
s = size(train_images);
o = ones(1,s(2));
train_images = [o;train_images];


train_labels = loadMNISTLabels(strcat(dir,'/train-labels-idx1-ubyte'));
test_images = loadMNISTImages(strcat(dir, '/t10k-images-idx3-ubyte'));
s = size(test_images);
o = ones(1,s(2));
test_images = [o;test_images];
test_labels = loadMNISTLabels(strcat(dir,'/t10k-labels-idx1-ubyte'));


end