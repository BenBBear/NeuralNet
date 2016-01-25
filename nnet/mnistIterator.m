classdef mnistIterator < handle
    %MNISTITERATORC Summary of this class goes here
    %   Detailed explanation goes here
    
    properties (GetAccess=public,SetAccess=private)
        trainCursor = 1; 
        testCursor = 1;
        trainIteration = 1;
        dataDimension = -1;
        trainSetSize = -1;  %% change in constructor
        testSetSize = -1;  %% change in constructor
    end
    properties (GetAccess=private,SetAccess=private)
        % Add in constructor
        train_images = [];
        train_labels = [];
        test_images = [];
        test_labels = [];
    end
    
    
    methods
        function this = mnistIterator(train_images, train_labels, test_images, test_labels)
            s = size(test_images);
            this.trainSetSize = length(train_labels);
            this.testSetSize = length(test_labels);
            this.dataDimension = s(1);
            this.train_images = train_images;
            this.train_labels = train_labels;
            this.test_images = test_images;
            this.test_labels = test_labels;
        end        
       function d = nextTrain(this)
            d = struct('data',this.train_images(:,this.trainCursor),...
                            'label',this.train_labels(this.trainCursor));         
            this.trainCursor = this.trainCursor+1;
            if this.trainCursor > this.trainSetSize
                this.trainCursor = 1;
                this.trainIteration = this.trainIteration+1;
            end        
       end
        
        function d = nextTest(this)
            if this.hasNextTest()
                d = struct('data', this.test_images(:,this.testCursor),...
                        'label', this.test_labels(this.testCursor));
                this.testCursor = this.testCursor+1;
            else
                error('Error. \nTest Set Stop Iterations. Please object.rewindTest() First.') 
            end
        end

        
        function output = hasNextTest(this)
            output = this.testCursor <= this.testSetSize;
        end

        function rewindTest(this)
            this.testCursor = 1;
        end        
    end
    
end

