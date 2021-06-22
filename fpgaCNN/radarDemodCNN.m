% radarDemodCNN.m
% Dylan Tocci
% UMass Dartmouth
% 6/22/2021
%
% This program trains (if necessary) a squeezenet CNN to demodulate
% signals based on their WVD image output. It should be noted a squeezenet
% was used as it applies square filters in convolution which is necessary
% for FPGA implementation. The CNN can determine modulation on the
% following signal types:
%   * B-FM      * GFSK
%   * Barker    * LFM
%   * CPFSK     * Rect
%   * DSB-AM    * SSB-AM
%
% The majority of this code was taken from the Mathworks example below:
% https://www.mathworks.com/help/phased/ug/modulation-classification-of-radar-and-communication-waveforms-using-deep-learning.html
% It has since been modified to allow test usage on the Xilinx Zynq706
%

%% Program parameters
trNet = false;      % Train neural network (only necessary if modified on .mat ...
                    % file is lost
genTestData = true; % Generate test data for FPGA / network testing
reprogram = false;  % Reprogram the Zynq706 board

%% Generate Data
rng default
[wav, modType] = helperGenerateRadarWaveforms();
parentDir = tempdir;
dataDir = 'TFDDatabase';
helperGenerateTFDfiles(parentDir,dataDir,wav,modType,200e3)
folders = fullfile(parentDir,dataDir,{'Rect','LFM','Barker','GFSK','CPFSK','B-FM','SSB-AM','DSB-AM'});
imds = imageDatastore(folders,...
    'FileExtensions','.png','LabelSource','foldernames','ReadFcn',@readTFDForSqueezeNet);
[imdsTrain,imdsTest,imdsValidation] = splitEachLabel(imds,0.8,0.1);

%% Train Network
if trNet
    trainSqueezeNetwork(imdsTrain, imdsValidation);
end

%% Load Network
net = load('demodSqueezeNet.mat');
snet = net.trainedNet;
% analyzeNetwork(snet)

%% Program FPGA
if reprogram
    hdlsetuptoolpath('ToolName', 'Xilinx Vivado', 'ToolPath', 'C:\Xilinx\Vivado\2019.1\bin\vivado.bat');
    hTarget = dlhdl.Target('Xilinx');

    hW = dlhdl.Workflow('Network', snet, 'Bitstream', 'zc706_single','Target',hTarget);
    dn = hW.compile

    hW.deploy
end

%% Test FPGA
% Grab random image
imgIndex = floor(1 + (length(imdsTest.Labels) - 1) * rand());
testImage = readimage(imdsTest,imgIndex);
imshow(testImage)
title(imdsTest.Labels(imgIndex))

[prediction, speed] = hW.predict(testImage,'Profile','on');
[val, idx] = max(prediction);

fprintf('The modulation was %s \nThe prediction result is %s\n', ...
        imdsTest.Labels(imgIndex), ...
        snet.Layers(end).ClassNames{idx});
