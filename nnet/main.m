if ~(exist('train_images','var') && exist('train_labels','var') && exist('test_images','var') && exist('test_labels','var'))
    [train_images, train_labels, test_images, test_labels]= initMnistData();
end

%%% TODO: add a forloop to loop through all kinds of activation function
%%%  - sigmod
%%%  - Tanh
%%%  - ReLU
configurations = {
    struct('learningRate',0.001,'hiddenNum',20,'maxIteration', 13,'stopErrorRate',0.001,'activationType','relu'),...
    struct('learningRate',0.001,'hiddenNum',20,'maxIteration', 3,'stopErrorRate',0.001,'activationType','relu'),...
    struct('learningRate',0.001,'hiddenNum',20,'maxIteration', 3,'stopErrorRate',0.001,'activationType','relu')
    };


for i = 1:length(configurations)
    config = configurations{i};
    iterator = mnistIterator(train_images, train_labels, test_images, test_labels);    
    model = struct('iterator',iterator,'config',config,'net', nnet(iterator,config.hiddenNum,10,config.learningRate,config.activationType));
    model.net.train(config.maxIteration, config.stopErrorRate);
    save(strcat('./result/myNet-',config.activationType,'.mat'),'model');
end

% iterator = mnistIterator(train_images, train_labels, test_images, test_labels);
% myNet = nnet(iterator,20,10,0.001,'sigmod');
% myNet.train(120,0.001)
% save('./result/myNet.mat','myNet','iterator');