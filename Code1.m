% Load your data
data = readtable('your_data.csv'); % Replace with your data source
% Separate features 
numerical_features = data2(:, 1:2);
categorical_features = data2(:, 3:5);
out=data2(:,6)

%% % Convert categorical variables to dummy variables (one-hot encoding)
% One-hot encoding for concreteOverlay
concreteOverlay_dummy = dummyvar(categorical_features.concreteOverlay);
% One-hot encoding for concreteOverlay
testType_dummy = dummyvar(categorical_features.testType);
% One-hot encoding for concreteOverlay
surfaceRoughness_dummy = dummyvar(categorical_features.surfaceRoughness);

%%%% Create variable names for the one-hot encoded columns
concreteOverlay_varnames = strcat('concreteOverlay_', string(categories(categorical_features.concreteOverlay)'));
testType_varnames = strcat('testType_', string(categories(categorical_features.testType))');
surfaceRoughness_varnames = strcat('surfaceRoughness_', string(categories(categorical_features.surfaceRoughness))');

%%% Convert dummy variables to table
concreteOverlay_table = array2table(concreteOverlay_dummy, 'VariableNames', concreteOverlay_varnames);
testType_table = array2table(testType_dummy, 'VariableNames', testType_varnames);
surfaceRoughness_table = array2table(surfaceRoughness_dummy, 'VariableNames', surfaceRoughness_varnames);
% Combine all tables
encoded_data = [testType_table,surfaceRoughness_table, concreteOverlay_table];
disp(encoded_data);

%%  Perform Min-Max Normalization for numerical_features
% Check the data type
disp(class(numerical_features));
if istable(numerical_features)
    numerical_features = table2array(numerical_features);
end
normalized_numerical_features_minmax = (numerical_features - min(numerical_features, [], 1)) ./ (max(numerical_features, [], 1) - min(numerical_features, [], 1));
disp(class(out));
if istable(out)
    out = table2array(out);
end
y1min=min(out(:,1));
y1max=max(out(:,1));
out1_norm=(out(:,1)-y1min)/(y1max-y1min);
out_norm=[out1_norm]
o= out_norm;
%% % Combine all features back into a single table in this step the all features must be numeric array
% Convert encoded_data to numeric array if it is a table
if istable(encoded_data)
    encoded_data = table2array(encoded_data);
end
data=[encoded_data, normalized_numerical_features_minmax];
disp(data)

%% %%build artificial neural network
input=data(:,1:13);
output=o;

inputs =  input';
targets = o';

% Create a Fitting Network
hiddenLayer1Size = [10];
net = fitnet(hiddenLayer1Size);


% Choose Input and Output Pre/Post-Processing Functions
% For a list of all processing functions type: help nnprocess
net.inputs{1}.processFcns = {'removeconstantrows','mapminmax'};
net.outputs{2}.processFcns = {'removeconstantrows','mapminmax'};


% Setup Division of Data for Training, Validation, Testing
% For a list of all data division functions type: help nndivide
net.divideFcn = 'dividerand';  % Divide data randomly
net.divideMode = 'sample';  % Divide up every sample
net.divideParam.trainRatio = 75/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio =10/100;

% For help on training function 'trainlm' type: help trainlm
% For a list of all training functions type: help nntrain
net.trainFcn = 'trainlm';  % Levenberg-Marquardt

% Choose a Performance Function
% For a list of all performance functions type: help nnperformance
net.performFcn = 'mse';  % Mean square error

% Choose Plot Functions
% For a list of all plot functions type: help nnplot
net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
  'plotregression', 'plotfit'};


% Train the Network
[net,tr] = train(net,inputs,targets);

% Test the Network
outputs = net(inputs);
errors = gsubtract(targets,outputs);
performance = perform(net,targets,outputs)

% Recalculate Training, Validation and Test Performance
trainTargets = targets .* tr.trainMask{1};
valTargets = targets  .* tr.valMask{1};
testTargets = targets  .* tr.testMask{1};
trainPerformance = perform(net,trainTargets,outputs)
valPerformance = perform(net,valTargets,outputs)
testPerformance = perform(net,testTargets,outputs)

% View the Network
view(net)

% Plots
% Uncomment these lines to enable various plots.
%figure, plotperform(tr)
%figure, plottrainstate(tr)
%figure, plotfit(net,inputs,targets)
%figure, plotregression(targets,outputs)
%figure, ploterrhist(errors)
%% determinde the ANN sensitivity

% Step 1: Get weights
weightsInputToHidden = net.IW{1}; % Weights from input to hidden layer
weightsHiddenToOutput = net.LW{2,1}; % Weights from hidden to output layer

% Step 2: Get dimensions
numInputs = size(weightsInputToHidden, 2); % Number of input parameters
numHidden = size(weightsInputToHidden, 1); % Number of hidden neurons
numOutputs = size(weightsHiddenToOutput, 1); % Number of output neurons

% Debugging: Display sizes of weights
disp('Weights from Input to Hidden:');
disp(size(weightsInputToHidden)); % Expected: [numHidden, numInputs]
disp('Weights from Hidden to Output:');
disp(size(weightsHiddenToOutput)); % Expected: [numOutputs, numHidden]
disp(['Number of Inputs: ', num2str(numInputs)]);
disp(['Number of Hidden Neurons: ', num2str(numHidden)]);
disp(['Number of Outputs: ', num2str(numOutputs)]);

% Step 3: Calculate absolute weights
absWeightsInputToHidden = abs(weightsInputToHidden); % |w_ij|
absWeightsHiddenToOutput = abs(weightsHiddenToOutput); % |w_jk|

% Step 4: Calculate the contribution Q_ik
Q = zeros(numInputs, numOutputs); % Initialize contributions matrix

for k = 1:numOutputs
    for i = 1:numInputs
        % Calculate the sum of absolute weights from input i to each hidden neuron
        sumAbsWeightsHidden = sum(absWeightsInputToHidden(:, i));
        
        % If the sum is not zero, proceed
        if sumAbsWeightsHidden ~= 0
            % Calculate Q_ik
            for j = 1:numHidden
                % Ensure we are accessing valid indices
                if j <= numHidden && k <= numOutputs
                    Q(i, k) = Q(i, k) + (absWeightsInputToHidden(j, i) / sumAbsWeightsHidden) * absWeightsHiddenToOutput(k, j);
                else
                    disp(['Index error: j = ', num2str(j), ', k = ', num2str(k)]);
                end
            end
        end
    end
    
    % Normalize contributions for output k
    totalContribution = sum(Q(:, k)); % Total contribution for output k
    if totalContribution ~= 0
        Q(:, k) = Q(:, k) / totalContribution; % Normalize
    end
end

% Step 5: Display contributions
for k = 1:numOutputs
    for i = 1:numInputs
        fprintf('Contribution of Input %d to Output %d: %.4f\n', i, k, Q(i, k));
    end
end
%% prediction
test = readtable('test.csv'); % Replace with your data source
% Separate features 
numerica2_features = test(:, 4:5);
categorica2_features = test(:, 1:3);
%% % Convert categorical variables to dummy variables (one-hot encoding)
% One-hot encoding for concreteOverlay
concreteOverlay_dummy = dummyvar(categorica2_features.concreteOverlay);
% One-hot encoding for concreteOverlay
testType_dummy = dummyvar(categorica2_features.testType);
% One-hot encoding for concreteOverlay
surfaceRoughness_dummy = dummyvar(categorica2_features.surfaceRoughness);

%%%% Create variable names for the one-hot encoded columns
concreteOverlay_varnames = strcat('concreteOverlay_', string(categories(categorica2_features.concreteOverlay)'));
testType_varnames = strcat('testType_', string(categories(categorica2_features.testType))');
surfaceRoughness_varnames = strcat('surfaceRoughness_', string(categories(categorica2_features.surfaceRoughness))');

%%% Convert dummy variables to table
concreteOverlay_table = array2table(concreteOverlay_dummy, 'VariableNames', concreteOverlay_varnames);
testType_table = array2table(testType_dummy, 'VariableNames', testType_varnames);
surfaceRoughness_table = array2table(surfaceRoughness_dummy, 'VariableNames', surfaceRoughness_varnames);
% Combine all tables
encoded1_data = [testType_table,surfaceRoughness_table, concreteOverlay_table];
disp(encoded_data);

%%  Perform Min-Max Normalization for numerical_features
% Check the data type
disp(class(numerica2_features));
if istable(numerica2_features)
    numerica2_features = table2array(numerica2_features);
end
normalized_numerica2_features_minmax = (numerica2_features - min(numerica2_features, [], 1)) ./ (max(numerica2_features, [], 1) - min(numerica2_features, [], 1));
%% % Combine all features back into a single table in this step the all features must be numeric array
% Convert encoded_data to numeric array if it is a table
if istable(encoded1_data)
    encoded1_data = table2array(encoded1_data);
end
data_test = [encoded1_data, normalized_numerica2_features_minmax];
disp(data_test)
ytest = net(data_test')';

ytest_denorm1=(ytest(:,1)*(y1max-y1min)+y1min);
ytest_denorm=[ ytest_denorm1];
disp(ytest_denorm)
%%
% Extract weights and biases from input to hidden layer
W1 = net.IW{1, 1}; % Weights from input to hidden layer
b1 = net.b{1};     % Biases for hidden layer

% Extract weights and biases from hidden to output layer
W2 = net.LW{2, 1}'; % Weights from hidden to output layer
b2 = net.b{2};      % Biases for output layer

% Display the weights and biases
disp('Weights from Input to Hidden Layer:');
disp(W1);
disp('Biases for Hidden Layer:');
disp(b1);

disp('Weights from Hidden to Output Layer:');
disp(W2);
disp('Biases for Output Layer:');
disp(b2);