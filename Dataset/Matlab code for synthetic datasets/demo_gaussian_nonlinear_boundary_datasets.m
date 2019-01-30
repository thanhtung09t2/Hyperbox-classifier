Mu_1 = [-2, 1.5];
Mu_2 = [1.5, 1];
Mu_3 = [-1.5, 3];
Mu_4 = [1.5, 2.5];

sigma_1 = [0.5, 0.05; 0.05, 0.4];
sigma_2 = [0.5, 0.05; 0.05, 0.3];
sigma_3 = [0.5, 0; 0, 0.5];
sigma_4 = [0.5, 0.05; 0.05, 0.2];

test_class{1} = mvnrnd(Mu_1, sigma_1, 2500);
test_class{1} = [test_class{1}; mvnrnd(Mu_2, sigma_2, 2500)];
test_class{2} = mvnrnd(Mu_3, sigma_3, 2500);
test_class{2} = [test_class{2}; mvnrnd(Mu_4, sigma_4, 2500)];

figure
plot(test_class{1}(:, 1),test_class{1}(:, 2),'b*');
hold on;
plot(test_class{2}(:, 1),test_class{2}(:, 2),'r+');