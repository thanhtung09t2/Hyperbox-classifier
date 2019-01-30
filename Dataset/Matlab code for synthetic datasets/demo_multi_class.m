N_sample = 10000;
numClasses = 5;
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

color = ['r','g','b','y','k', 'm', 'c', 'r', 'g', 'b', 'y'];
marker = ['o', '+', '*', '^', 'd', 's', 'v', '.', '<', '>', '*'];

% Specific a covariance matrix
D = 2;
SIGMA = zeros(D, D);
for i = 1:D
    SIGMA(i, i) = 1;
end

for i = 1:numClasses
    mu_i = [2.56, 2.56] .* corTotal{i};
    data{i} = mvnrnd(mu_i, SIGMA, int64(N_sample / numClasses));
end

if numClasses < 12
    figure;
    hold on;
    for i = 1:numClasses
        scatter(data{i}(:, 1), data{i}(:, 2),'MarkerEdgeColor',color(i), 'Marker', marker(i));
    end
end

