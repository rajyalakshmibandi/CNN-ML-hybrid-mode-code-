% Set path to the BUSI dataset
datasetPath = 'C:datasets path';  % <-- Adjust if needed

% Define custom ReadFcn to convert grayscale to RGB
customReadFcn = @(filename) ensureRGB(imread(filename));

% Create imageDatastore with custom ReadFcn
imds = imageDatastore(datasetPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames', ...
    'ReadFcn', customReadFcn);

% Split data: 70% train, 30% test
[imdsTrain, imdsTest] = splitEachLabel(imds, 0.7, 'randomized');

%Replace the layers
net=inceptionresnetv2;
analyzeNetwork(net)

% Input size for inceptionresnetv2
inputSize = net.Layers(1).InputSize;

% Feature extraction layer
featureLayer = 'avg_pool';  % Last fully connected layer before classification

% Preprocess and resize images
augTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, 'ColorPreprocessing', 'gray2rgb');
augTest = augmentedImageDatastore(inputSize(1:2), imdsTest, 'ColorPreprocessing', 'gray2rgb');

% Extract features from training data
featuresTrain = activations(net, augTrain, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'rows');
labelsTrain = imdsTrain.Labels;

% Train SVM classifier
classifier = fitcecoc(featuresTrain, labelsTrain);

% Extract features from test data
featuresTest = activations(net, augTest, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'rows');
labelsTest = imdsTest.Labels;

% Predict labels for test set
predictedLabels = predict(classifier, featuresTest);

% Calculate accuracy
accuracy = mean(predictedLabels == labelsTest);
fprintf('Test Accuracy (inceptionresnetv2): %.2f%%\n', accuracy * 100);

% Plot confusion matrix
figure;
confusionchart(labelsTest, predictedLabels);
%title('Confusion Matrix - VGG-19+SVM');
title('Confusion Matrix - inceptionresnetv2+SVM');

% -------- Function to ensure RGB format --------
function imgRGB = ensureRGB(img)
    if ndims(img) == 2  % Grayscale
        imgRGB = cat(3, img, img, img);
    elseif ndims(img) == 3 && size(img, 3) == 1
        imgRGB = cat(3, img(:,:,1), img(:,:,1), img(:,:,1));
    elseif ndims(img) == 3 && size(img, 3) == 3
        imgRGB = img;
    else
        error('Unexpected image format');
    end

    % Convert to uint8 if needed
    if ~isa(imgRGB, 'uint8')
        imgRGB = im2uint8(imgRGB);
    end
end
