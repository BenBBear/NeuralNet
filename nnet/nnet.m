classdef nnet < handle
    %NNET Summary of this class goes here
    %   Detailed explanation goes here
    %   A Three Layers Neural Network
    %   Input | Hidden | Output
    properties(GetAccess=public,SetAccess=public)
        console = true;
        verbose = true;
        iterator;
        %%%%%
        fields;
        %%%%%
        layerDimension% cell Array
        %%%%%  
        W % cell Array
        Delta; % cell Array
        Value;
        %%%%%
        target;
        %%%%%
        layerNum;       
        %%%%%
        expectedLabel;
        outputLabel;        
        %%%%%
        errorNum = 0;
        errRate = Inf;   
        %%%%%
        learningRate = 0.001;
        activationType = 'sigmod';
        finalLayerType = 'softmax';
        batchNum = 1;
        regularizationType = 'none'; %%none,l1,l2
        regularizationLambda = 0.01;
        
        %%%%
        currentTrainingLayer;
        batchErrorNum = 0;
        
        %%%%
        useMomentum = false;
        momentumCoef = 0.3;
        Velocity;
        iterationNum = 0;
        errorFunctionType = 'crossEntrophy'
        epsilon = 10^-5;
    end
    
    methods                 
        function this = nnet(option)
            this.fields = {'console', 'iterator','finalLayerType','useMomentum','momentumCoef'...
                           'layerDimension','regularizationType','regularizationLambda',...
                           'epsilon',...
                           'learningRate', 'activationType', 'batchNum','verbose','errorFunctionType'};
            
            for i = 1:length(this.fields)
                try
                    setfield(this,this.fields{i}, ...
                        getfield(option, this.fields{i}));
                catch 
                end
            end
            this.layerNum = length(this.layerDimension);
            this.W = cell(1,this.layerNum-1);
            this.Delta = cell(this.batchNum, this.layerNum-1);
            this.Value = cell(this.batchNum, this.layerNum);
            if this.useMomentum
                this.Velocity = cell(1,this.layerNum-1);
                for i = 1:this.layerNum-1
                    this.Velocity{i} = 0;
                end
            end
            this.expectedLabel = zeros(1,this.batchNum); 
            this.outputLabel = zeros(1,this.batchNum); 
            this.target = cell(1,this.batchNum);
            for i = 1:this.layerNum-1
               this.W{i} = rand(this.layerDimension(i),...
                                this.layerDimension(i+1));               
            end
            this.currentTrainingLayer = this.layerNum-1;
            if this.layerNum < 2
                error('Error.\nLayer Number should be larger than 2.');
            end  
            if this.batchNum > this.iterator.trainSetSize
                 error('Error.\nBatch Number should be less than train set size.');
            end
        end
        
        
        function d = g(this,x)     
            switch(this.activationType)
                case 'sigmod'
                    d = sigmf(x,[1 0]);  
                case 'tanh'
                    d = tanh(x);
                case 'relu'                 
                    d = max(x,0);
                otherwise
                    d = sigmf(x,[1 0]);
            end
        end
        
        function d = dg(this,x)
            switch(this.activationType)
                case 'sigmod'
                   d = x.*(1-x);
                case 'tanh'
                   d = 1 - x.^2;
                case 'relu'
                   d = x>0;
                otherwise
                   d = x.*(1-x);
            end  
        end                        
        function d = final_g(this,x)
            switch(this.finalLayerType)
                case 'softmax'
                    d = exp(x)./sum(exp(x));
                otherwise
                    d = exp(x)./sum(exp(x));
            end            
        end
        
        
        function fetchTrainData(this)
            iold = this.iterator.trainIteration;                        
            for i = 1:this.batchNum
                x = this.iterator.nextTrain();
                this.Value{i,1} = x.data';
                this.expectedLabel(i) = x.label;
                this_target = zeros(1,this.layerDimension(this.layerNum));
                this_target(x.label) = 1;
                this.target{i} = this_target;
            end
            inew = this.iterator.trainIteration;
            if inew > iold
               this.trainErrorSummary(iold); 
            end
        end   
        
        function trainErrorSummary(this,iold)
            this.errRate = this.errorNum/this.iterator.trainSetSize;
            if this.console
                str = sprintf('Iteration No %d, Train Data size: %d, Error rate %f, Error Num is %d',iold, this.iterator.trainSetSize, this.errRate, this.errorNum);
                disp(str);
            end
            this.errorNum = 0;
        end
        function forward(this)
            for j = 1:this.batchNum
                for i = 1:this.layerNum-2
                    this.Value{j,i+1} = this.g(this.Value{j,i}*this.W{i});
                end
                this.Value{j,this.layerNum} = this.final_g(this.Value{j,this.layerNum-1}*this.W{this.layerNum-1});
                [~,lb] = max(this.Value{j,this.layerNum});
                this.outputLabel(j) = lb;
            end
        end
        function recordError(this)
            this.batchErrorNum = 0;
            for i = 1:this.batchNum
                if this.expectedLabel(i) ~= this.outputLabel(i)
                    this.errorNum = this.errorNum + 1;
                    this.batchErrorNum = this.batchErrorNum+1;
                end
            end 
            if this.verbose && this.console
                str = sprintf('Batch Size: %d, Error Number: %d, Error Rate: %f', ...
                    this.batchNum,this.batchErrorNum, this.batchErrorNum/this.batchNum);
                disp(str);
            end
        end        
        function lb = classify(this,input)
            for i = 1:this.layerNum-2
                input = this.g(input*this.W{i});
            end
            input = this.final_g(input*this.W{this.layerNum-1});
            [~,lb] = max(input);         
        end
        function softmaxFinalLayer(this)
             for i = 1:this.batchNum
                 this.Delta{i,this.currentTrainingLayer} = this.Value{i,this.currentTrainingLayer+1} - this.target{i};
             end 
        end
        
        function updateDeltaFinalLayer(this)
            switch(this.finalLayerType)
                case 'softmax'
                     this.softmaxFinalLayer();                
                otherwise
                     this.softmaxFinalLayer();
            end               
        end
        function updateDeltaMiddleLayer(this)
            for i = 1:this.batchNum
                this.Delta{i,this.currentTrainingLayer} = this.dg(this.Value{i,this.currentTrainingLayer+1}).*...
                                                        (this.Delta{i,this.currentTrainingLayer+1} * ...
                                                        this.W{this.currentTrainingLayer+1}');
            end
        end
        function dW = gradient(this)
            dW = 0;
            for i=1:this.batchNum
                dW = dW + this.Value{i,this.currentTrainingLayer}'*this.Delta{i,this.currentTrainingLayer};
            end
            dW = dW/this.batchNum;
        end
        function r = regularization(this)
            switch(this.regularizationType)
                case 'none'
                    r = 0;
                case 'l1'
                    r = sign(this.W{this.currentTrainingLayer})*this.regularizationLambda/this.batchNum;
                case 'l2'
                    r = this.W{this.currentTrainingLayer}*this.regularizationLambda/this.batchNum;
                otherwise
                    r = 0;
            end
        end
        function updateDelta(this)
            if this.currentTrainingLayer == this.layerNum - 1
                this.updateDeltaFinalLayer();
            else
                this.updateDeltaMiddleLayer();
            end
        end
        
        function gd = updateOnce(this)            
            this.updateDelta();
            gd = this.gradient();
            if this.useMomentum
                this.Velocity{this.currentTrainingLayer} = this.Velocity{this.currentTrainingLayer}*this.momentumCoef - ...
                                                            this.learningRate * (gd+this.regularization());
                this.W{this.currentTrainingLayer} = this.W{this.currentTrainingLayer} + ...
                                                      this.Velocity{this.currentTrainingLayer};
            else
                this.W{this.currentTrainingLayer} = this.W{this.currentTrainingLayer} - ...
                                        this.learningRate * (this.regularization()+gd);
            end                                                                        
        end
        
        function update(this)
           while this.currentTrainingLayer >= 1
               this.updateOnce();
               this.currentTrainingLayer = this.currentTrainingLayer - 1;
           end           
        end
                  
        function resetCurrentLayer(this)
            this.currentTrainingLayer = this.layerNum - 1;
        end
        function backward(this)
            this.update();
            this.resetCurrentLayer();
        end
        
        function iter(this)
            this.iterationNum = this.iterationNum + 1;
            this.fetchTrainData();
            this.forward();
            this.recordError();
            this.backward();          
        end
        
        function train(this,iterTimes, errRateThreshold)
            this.resetCurrentLayer();
            while this.iterationNum <= iterTimes && ...                     
                    this.errRate > errRateThreshold
                this.iter()
            end
            str = sprintf('Training is finished, current Error rate is %0.3f',this.errRate);
            disp(str);
        end                
        function er = test(this)
            testSize = this.iterator.testSize;
            errorNumber = 0;
            while this.iterator.hasNextTest()
                x = this.iterator.nextTest();
                lb = this.classify(x.data');
                if lb ~= x.label
                    errorNumber = errorNumber + 1;
                end
            end
            er = errorNumber/testSize;
            str = sprintf('Test on %d of datas, with Error Rate: %0.3f', testSize, er);
            disp(str);
        end
        
        function [ dW,dWe,analysis ] = gradientCheck(this)
           e = this.epsilon;
           analysis = cell(1,this.layerNum-1);
           dW = cell(1,this.layerNum-1);
           dWe = cell(1,this.layerNum-1);
           this.resetCurrentLayer();
           this.fetchTrainData();           
           for l = 1:this.layerNum-1
               mn = size(this.W{l});              
               dWe{l} = zeros(mn);
               for x = 1:mn(1)
                   for y = 1:mn(2)
                       this.W{l}(x,y) = this.W{l}(x,y) + e;
                       this.forward();
                       e1 = this.E();
                       this.W{l}(x,y) = this.W{l}(x,y) - 2*e;
                       this.forward();
                       e2 = this.E();
                       dWe{l}(x,y) = (e1-e2)/(2*e);
                       this.W{l}(x,y) = this.W{l}(x,y) + e; %% reset
                   end
               end           
           end           
           while this.currentTrainingLayer >= 1
               dW{this.currentTrainingLayer} = this.updateOnce();
               this.currentTrainingLayer = this.currentTrainingLayer - 1;
           end
           for k = 1:this.layerNum-1
               analysis{k} = struct('mean',this.gradientCheckMean(dW{k},dWe{k}),...
                                    'max',this.gradientCheckMax(dW{k},dWe{k}));
           end
        end
        function r = gradientCheckMean(this,m1,m2)
            r = mean(abs(mean(abs(m1-m2))));
        end
        function r = gradientCheckMax(this,m1,m2)
            r = max(abs(max(abs(m1-m2))));
        end
        function d = E(this)
            switch(this.errorFunctionType)
                case 'crossEntrophy'
                    d = this.crossEntrophyE();
                otherwise
                    d = this.crossEntrophyE();
            end            
        end
        function r = crossEntrophyE(this)
            %%% this.target
            r = 0;
            for k = 1:this.batchNum
                t = this.target{k};
                r = r + sum(t.*log(this.Value{k,this.layerNum}));
            end
            r = -r/k;
        end
    end  
end

