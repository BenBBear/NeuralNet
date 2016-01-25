if ~(exist('train_images','var') && exist('train_labels','var') && exist('test_images','var') && exist('test_labels','var'))
    [train_images, train_labels, test_images, test_labels]= initMnistData();
end

limit = 2000;
iterator = mnistIterator(train_images(:,1:2000), train_labels(1:2000)+1, test_images, test_labels+1);

net = nnet(struct('iterator',iterator,...
    'layerDimension',[785,20,10], ...
    'batchNum',1,'regularizationType','none',...
    'verbose',false,'useMomentum',true,'momentumCoef',0,'activationType','sigmod'));

%%net.train(30000,0.1);

[dW,dWe,comparision] = net.gradientCheck();
