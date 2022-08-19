clear all
load('TwoDmanupImages');
XTrain=double(imgObjStore(:,:,1,1:500));
XValidation=double(imgObjStore(:,:,1,501:1000));

YTrain=round(thetaList(1:500,:));
YValidation=round(thetaList(501:1000,:));

 

 layers = [
    imageInputLayer([50 50 1])
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    averagePooling2dLayer(2,'Stride',2)
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    averagePooling2dLayer(2,'Stride',2)
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.2)
    fullyConnectedLayer(2)
    regressionLayer];


  miniBatchSize  = 10;
  validationFrequency = floor(numel(YTrain)/miniBatchSize);
  options = trainingOptions('sgdm', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',30, ...
    'InitialLearnRate',1e-3, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.1, ...
    'LearnRateDropPeriod',20, ...
    'Shuffle','every-epoch', ...
    'ValidationData',{XValidation,YValidation}, ...
    'ValidationFrequency',validationFrequency, ...
    'Plots','training-progress', ...
    'Verbose',false);  

  net = trainNetwork(XTrain,YTrain,layers,options);