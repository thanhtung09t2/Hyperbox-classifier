numClasses = 25;
thresValStep = 0;
xCor = 0;
yCor = 0;
minValStep = 0;
corTotal = {};
for c = 1:numClasses
    if xCor + yCor == 2 * thresValStep
        corTotal{c} = [xCor, yCor];
        thresValStep = thresValStep + 1;
        xCor = thresValStep;
        yCor = 0;
        minValStep = 0;
    elseif xCor == thresValStep
        corTotal{c} = [xCor, yCor];
        yCor = xCor;
        xCor = minValStep;
        minValStep = minValStep + 1;
    else
        corTotal{c} = [xCor, yCor];
        xCor = thresValStep;
        yCor = minValStep;
    end
end
for c = 1:numClasses
    disp(corTotal{c})
end

D = 2; % Number of dimension
%N_tr = [10000, 50000, 100000, 500000, 1000000, 5000000, 10000000]; % Number of training samples
N_tr = [10000, 50000, 100000, 500000, 1000000, 5000000]; % Number of training samples
N_val = 10000; % Number of validation samples
N_test = 100000; % Number of testing samples

%pre_file_training_name = ["10K", "50K", "100K", "500K", "1M", "5M", "10M"];
pre_file_training_name = ["10K", "50K", "100K", "500K", "1M", "5M"];

for i = 1:size(N_tr, 2)
    training_file_name(1, i) = 'N' + pre_file_training_name(1, i) + '-D-' + string(D) + '-C-' + string(numClasses) + '_train.dat';
end

testing_file_name = 'N' + string(N_test) + '-D-' + string(D) + '-C-' + string(numClasses) + '_test.dat';
val_file_name = 'N' + string(N_val) + '-D-' + string(D) + '-C-' + string(numClasses) + '_val.dat';
para_file_name = 'N' + string(N_test) + '-D-' + string(D) + '-C-' + string(numClasses) + '_para.dat';
% Specific a covariance matrix
SIGMA = zeros(D, D);
for i = 1:D
    SIGMA(i, i) = 1;
end

% Generate bivariate normal distributions with specified means for training
% data
training_class = {};
min_training = [];
max_training = [];
for i = 1:size(N_tr, 2)
    for j = 1:numClasses
        mu_i = [2.56, 2.56] .* corTotal{j};
        training_class{i}{j} = mvnrnd(mu_i, SIGMA, int64(N_tr(1, i) / numClasses));
    end
end

for j = 1:D
    for i = 1:size(N_tr, 2)
        min_c = min(training_class{i}{1}(:, j));
        max_c = max(training_class{i}{1}(:, j));
        for cc = 2:numClasses
            if min_c > min(training_class{i}{cc}(:, j))
                min_c = min(training_class{i}{cc}(:, j));
            end
            if max_c < max(training_class{i}{cc}(:, j))
                max_c = max(training_class{i}{cc}(:, j));
            end
        end
        min_training(i, j) = min_c;
        max_training(i, j) = max_c;
    end
end

disp('min_training = ')
disp(min_training)
disp('max_training = ')
disp(max_training)

tr_min = min(min_training, [], 1);
tr_max = max(max_training, [], 1);

% Generate validation data
val_class = {};
min_val = [];
max_val = [];
for j = 1:numClasses
    mu_i = [2.56, 2.56] .* corTotal{j};
    val_class{j} = mvnrnd(mu_i, SIGMA, int64(N_val / numClasses));
end

for j = 1:D
    min_c = min(val_class{1}(:, j));
    max_c = max(val_class{1}(:, j));
    
    for cc = 2:numClasses
        if min_c > min(val_class{cc}(:, j))
            min_c = min(val_class{cc}(:, j));
        end
        if max_c < max(val_class{cc}(:, j))
            max_c = max(val_class{cc}(:, j));
        end
    end
    
    min_val(1, j) = min_c;
    max_val(1, j) = max_c;
end

% Generate testing data
test_class = {};
min_test = [];
max_test = [];
for j = 1:numClasses
    mu_i = [2.56, 2.56] .* corTotal{j};
    test_class{j} = mvnrnd(mu_i, SIGMA, int64(N_test / numClasses));
end

for j = 1:D
    min_c = min(test_class{1}(:, j));
    max_c = max(test_class{1}(:, j));
    
    for cc = 2:numClasses
        if min_c > min(test_class{cc}(:, j))
            min_c = min(test_class{cc}(:, j));
        end
        if max_c < max(test_class{cc}(:, j))
            max_c = max(test_class{cc}(:, j));
        end
    end
    
    min_test(1, j) = min_c;
    max_test(1, j) = max_c;
end

disp('tr_min =');
disp(tr_min);
disp('tr_max =');
disp(tr_max);
disp('min_val = ');
disp(min_val);
disp('max_val = ');
disp(max_val);
disp('min_test = ');
disp(min_test);
disp('max_test = ');
disp(max_test);
% Normalize data to [0, 1]
min_total = [];
max_total = [];
for i = 1:D
    min_total(1, i) = min(min(tr_min(i), min_val(i)), min_test(i));
    max_total(1, i) = max(max(tr_max(i), max_val(i)), max_test(i));
end

disp('min_total = ');
disp(min_total);
disp('max_total = ');
disp(max_total);

b = [0, 1];
maxValRange = [];
minValRange = [];
for ii=1:D
    minv = min_total(ii);
    maxv = max_total(ii);
    
    maxValRange = [maxValRange, maxv];
    minValRange = [minValRange, minv];
    
    for j = 1:size(N_tr, 2)
        for k = 1:numClasses
            v_tr_k = b(1) + (b(2)-b(1))*(training_class{j}{k}(:, ii)-minv) / (maxv-minv);
            if max(v_tr_k) > 1
                disp('class=')
                disp(k)
                disp('dataset index')
                disp(j)
                disp('dimension')
                disp(ii)
            end
            training_class{j}{k}(:,ii) = v_tr_k;
        end
    end
    
    for k = 1:numClasses
        v_val_k = b(1) + (b(2)-b(1))*(val_class{k}(:, ii)-minv) / (maxv-minv);
        val_class{k}(:,ii) = v_val_k;
    end
    
    for k = 1:numClasses
        v_test_k = b(1) + (b(2)-b(1))*(test_class{k}(:, ii)-minv) / (maxv-minv);
        test_class{k}(:,ii) = v_test_k;
    end
end

% Create datasets to save to file
label_tr = {};
for i = 1:size(N_tr, 2)
    disp(i)
    disp(N_tr(1, i))
    step = int64(N_tr(1, i) / numClasses);
    disp(step)
    for j = 1:numClasses
        label_tr{i}(((j-1) * step + 1): j * step) = j;
    end
    
    % create dataset
    train_data = [];
    for j = 1:numClasses
        train_data = [train_data; training_class{i}{j}];
    end
    disp(size(train_data))
    disp(size(label_tr{i}))
    train_data = [train_data, label_tr{i}'];
    
    disp(size(train_data))
    disp(training_file_name(1, i))
    dlmwrite(training_file_name(1, i), train_data);
end

step = int64(N_val / numClasses);
label_val = [];
for j = 1:numClasses
    label_val(((j-1) * step + 1): j * step) = j;
end

val_data = [];
for j = 1:numClasses
    val_data = [val_data; val_class{j}];
end
val_data = [val_data, label_val'];

step = int64(N_test / numClasses);
label_test = [];
for j = 1:numClasses
    label_test(((j-1) * step + 1): j * step) = j;
end

test_data = [];
for j = 1:numClasses
    test_data = [test_data; test_class{j}];
end
test_data = [test_data, label_test'];

dlmwrite(testing_file_name, test_data);
dlmwrite(val_file_name, val_data);

para = [maxValRange; minValRange];
dlmwrite(para_file_name, para);

if numClasses < 12
    color = ['r','g','b','y','k', 'm', 'c', 'r', 'g', 'b', 'y'];
    marker = ['o', '+', '*', '^', 'd', 's', 'v', '.', '<', '>', '*'];
    figure; hold on;
    for i = 1:numClasses
        scatter(training_class{1}{i}(:, 1), training_class{1}{i}(:, 2),'MarkerEdgeColor',color(i), 'Marker', marker(i));
    end
end