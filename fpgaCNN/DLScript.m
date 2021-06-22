%% Deep Learning Signal Classification
% https://www.mathworks.com/help/phased/ug/modulation-classification-of-radar-and-communication-waveforms-using-deep-learning.html
format short; clc; %clear;

% Generate waveforms
rng default
[wav, modType] = helperGenerateCommsWaveforms();

% Graph Wigner-Ville Distributions
figure(2)
subplot(2,3,1)
wvd(wav{find(modType == "GFSK",1)},200e3,'smoothedPseudo')
axis square; colorbar off; title('GFSK')
subplot(2,3,2)
wvd(wav{find(modType == "CPFSK",1)},200e3,'smoothedPseudo')
axis square; colorbar off; title('CPFSK')
subplot(2,3,3)
wvd(wav{find(modType == "B-FM",1)},200e3,'smoothedPseudo')
axis square; colorbar off; title('B-FM')
subplot(2,3,4)
wvd(wav{find(modType == "SSB-AM",1)},200e3,'smoothedPseudo')
axis square; colorbar off; title('SSB-AM')
subplot(2,3,5)
wvd(wav{find(modType == "DSB-AM",1)},200e3,'smoothedPseudo')
axis square; colorbar off; title('DSB-AM')

% Save .png files for smoothed WVDs
parentDir = tempdir;
dataDir = 'TFDDatabase';
helperGenerateTFDfiles(parentDir,dataDir,wav,modType,200e3)
folders = fullfile(parentDir,dataDir,{'Rect','LFM','Barker','GFSK','CPFSK','B-FM','SSB-AM','DSB-AM'});
imds = imageDatastore(folders,...
    'FileExtensions','.png','LabelSource','foldernames','ReadFcn',@readTFDForSqueezeNet);

% Split data, 80% training - 10% testing - 10% validation
[imdsTrain,imdsTest,imdsValidation] = splitEachLabel(imds,0.8,0.1);




% Set up Deep Learning Network
net = squeezenet;

% Adjust to fit needs
lgraphSqz = layerGraph(net);

% Fix Dropout layer
tmpLayer = lgraphSqz.Layers(end-5);
newDropoutLayer = dropoutLayer(0.6,'Name','new_dropout');
lgraphSqz = replaceLayer(lgraphSqz,tmpLayer.Name,newDropoutLayer);

% Fix Convolution layer
numClasses = 8;
tmpLayer = lgraphSqz.Layers(end-4);
newLearnableLayer = convolution2dLayer(1,numClasses, ...
        'Name','new_conv', ...
        'WeightLearnRateFactor',20, ...
        'BiasLearnRateFactor',20);
lgraphSqz = replaceLayer(lgraphSqz,tmpLayer.Name,newLearnableLayer);

% Fix Classification layer
tmpLayer = lgraphSqz.Layers(end);
newClassLayer = classificationLayer('Name','new_classoutput');
lgraphSqz = replaceLayer(lgraphSqz,tmpLayer.Name,newClassLayer);

% Set up training options
options = trainingOptions('sgdm', ...
    'MiniBatchSize',128, ...
    'MaxEpochs',5, ...
    'InitialLearnRate',1e-3, ...
    'Shuffle','every-epoch', ...
    'Verbose',false, ...
    'Plots','training-progress',...
    'ValidationData',imdsValidation);

% Begin training :(
trainedNet = trainNetwork(imdsTrain,lgraphSqz,options);

% See if its any good (probably not but who knows)
predicted = classify(trainedNet,imdsTest);
figure
confusionchart(predicted,imdsTest.Labels,'Normalization','column-normalized')
