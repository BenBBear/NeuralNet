function [train_images, train_labels, test_images, test_labels] = initMnistData()

if nargin == 0
    dir = mfilename('fullpath');
    dir = dir(1:length(dir)-13);
end

    
try
    load(strcat(dir, '/mnistData.mat'));
catch
    [train_images, train_labels, test_images, test_labels] = readMnist();
    train_images = zscore(train_images);
    test_images = zscore(test_images);
    sf_train = randperm(length(train_labels));
    sf_test = randperm(length(test_labels));
    train_images = train_images(:,sf_train);
    train_labels = train_labels(sf_train);
    test_images = test_images(:,sf_test);
    test_labels = test_labels(sf_test);
    
    save(strcat(dir, '/mnistData.mat'), 'train_images', ...
        'train_labels', 'test_images', 'test_labels');
end

end