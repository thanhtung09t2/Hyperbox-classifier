% Ref to Fukunaga, K. 1990. Introduction to Statistical Pattern Recognition, 2nd edition. Boston: Academic Press
% Two-class dataset with Gaussian distribution. Minimum Bayes error = 10%
% Class 1: mu = (0,...,0), sigma = (1,...,1)
% Class 2: mu = (2.56,0,...,0), sigma = (1,...,1)
clear;
D = 64; % Number of dimension
%N_tr = [10000, 50000, 100000, 500000, 1000000, 5000000, 10000000]; % Number of training samples
N_tr = [10000, 50000, 100000, 500000, 1000000, 5000000]; % Number of training samples
N_val = 10000; % Number of validation samples
N_test = 100000; % Number of testing samples

%pre_file_training_name = ["10K", "50K", "100K", "500K", "1M", "5M", "10M"];
pre_file_training_name = ["10K", "50K", "100K", "500K", "1M", "5M"];

for i = 1:size(N_tr, 2)
    training_file_name(1, i) = 'N' + pre_file_training_name(1, i) + '-D-' + string(D) + '-C-2_train.dat';
end

testing_file_name = 'N' + string(N_test) + '-D-' + string(D) + '-C-2_test.dat';
val_file_name = 'N' + string(N_val) + '-D-' + string(D) + '-C-2_val.dat';
para_file_name = 'N' + string(N_val) + '-D-' + string(D) + '-C-2_para.dat';
% Specific a covariance matrix
SIGMA = zeros(D, D);
for i = 1:D
    SIGMA(i, i) = 1;
end

Mu_1 = zeros(1, D);
Mu_2 = zeros(1, D);
Mu_2(1,1) = 2.56;
training_class = {};
min_training = [];
max_training = [];
% Generate bivariate normal distributions with specified means for training
% data
for i = 1:size(N_tr, 2)
    training_class{i}{1} = mvnrnd(Mu_1, SIGMA, N_tr(1, i) / 2);
    training_class{i}{2} = mvnrnd(Mu_2, SIGMA, N_tr(1, i) / 2);

    for j = 1:D
        min_training(i, j) = min(min(training_class{i}{1}(:, j)), min(training_class{i}{2}(:, j)));
        max_training(i, j) = max(max(training_class{i}{1}(:, j)), max(training_class{i}{2}(:, j)));
    end
end

tr_min = min(min_training, [], 1);
tr_max = max(max_training, [], 1);

% Generate validation data
val_class = {};
min_val = [];
max_val = [];

val_class{1} = mvnrnd(Mu_1, SIGMA, N_val / 2);
val_class{2} = mvnrnd(Mu_2, SIGMA, N_val / 2);

for i = 1:D
    min_val(1, i) = min(min(val_class{1}(:, i)), min(val_class{2}(:, i)));
    max_val(1, i) = max(max(val_class{1}(:, i)), max(val_class{2}(:, i)));
end

% Generate testing data
test_class = {};
min_test = [];
max_test = [];

test_class{1} = mvnrnd(Mu_1, SIGMA, N_test / 2);
test_class{2} = mvnrnd(Mu_2, SIGMA, N_test / 2);

for i = 1:D
    min_test(1, i) = min(min(test_class{1}(:, i)), min(test_class{2}(:, i)));
    max_test(1, i) = max(max(test_class{1}(:, i)), max(test_class{2}(:, i)));
end

disp(tr_min);
disp(tr_min);
disp(min_val);
disp(max_val);
disp(min_test);
disp(max_test);
% Normalize data to [0, 1]
for i = 1:D
    min_total(1, i) = min(min(tr_min(1, i), min_val(1, i)), min_test(1, i));
    max_total(1, i) = max(max(tr_max(1, i), max_val(1, i)), max_test(1, i));
end

b = [0, 1];
maxValRange = [];
minValRange = [];
for ii=1:D
    minv = min_total(1, ii);
    maxv = max_total(1, ii);
    
    maxValRange = [maxValRange, maxv];
    minValRange = [minValRange, minv];
    
    for j = 1:size(N_tr, 2)
        v_tr_1 = b(1) + (b(2)-b(1))*(training_class{j}{1}(:, ii)-minv) / (maxv-minv);
        v_tr_2 = b(1) + (b(2)-b(1))*(training_class{j}{2}(:, ii)-minv) / (maxv-minv);

        training_class{j}{1}(:,ii) = v_tr_1;
        training_class{j}{2}(:,ii) = v_tr_2;
    end
    
    v_val_1 = b(1) + (b(2)-b(1))*(val_class{1}(:, ii)-minv) / (maxv-minv);
    v_val_2 = b(1) + (b(2)-b(1))*(val_class{2}(:, ii)-minv) / (maxv-minv);

    v_test_1 = b(1) + (b(2)-b(1))*(test_class{1}(:, ii)-minv) / (maxv-minv);
    v_test_2 = b(1) + (b(2)-b(1))*(test_class{2}(:, ii)-minv) / (maxv-minv);
    
    val_class{1}(:,ii) = v_val_1;
    val_class{2}(:,ii) = v_val_2;
    
    test_class{1}(:,ii) = v_test_1;
    test_class{2}(:,ii) = v_test_2;
end

% Create datasets to save to file
label_tr = {};
for i = 1:size(N_tr, 2)
    label_tr{i}(1:N_tr(1, i)/2) = 1;
    label_tr{i}(N_tr(1, i)/2 + 1:N_tr(1, i)) = 2;
    
    % create dataset
    train_data = [training_class{i}{1}; training_class{i}{2}];
    train_data = [train_data, label_tr{i}'];
    disp(size(train_data))
    disp(training_file_name(1, i))
    dlmwrite(training_file_name(1, i), train_data);
end

label_val = [];
label_test = [];
label_val(1:N_val/2) = 1;
label_val(N_val/2 + 1:N_val) = 2;
label_test(1:N_test/2) = 1;
label_test(N_test/2 + 1:N_test) = 2;

test_data = [test_class{1}; test_class{2}];
test_data = [test_data, label_test'];
val_data = [val_class{1}; val_class{2}];
val_data = [val_data, label_val'];

dlmwrite(testing_file_name, test_data);
dlmwrite(val_file_name, val_data);

para = [maxValRange; minValRange];
dlmwrite(para_file_name, para);
    
% Visualization of training data 10K samples
if D == 2
    plot(training_class{1}{1}(:, 1),training_class{1}{1}(:, 2),'b*');
    hold on;
    plot(training_class{1}{2}(:, 1),training_class{1}{2}(:, 2),'r+');
elseif D == 3
    plot3(training_class{1}{1}(:, 1),training_class{1}{1}(:, 2), training_class{1}{1}(:, 3), 'b*');
    hold on;
    plot3(training_class{1}{2}(:, 1),training_class{1}{2}(:, 2), training_class{1}{1}(:, 3), 'r+');
end
