D = 2;
Mu_1 = zeros(1, D);
Mu_2 = zeros(1, D);
Mu_2(1,1) = 2.56;

SIGMA = zeros(D, D);
for i = 1:D
    SIGMA(i, i) = 1;
end

test_class{1} = mvnrnd(Mu_1, SIGMA, 5000);
test_class{2} = mvnrnd(Mu_2, SIGMA, 5000);

figure
plot(test_class{1}(:, 1),test_class{1}(:, 2),'b*');
hold on;
plot(test_class{2}(:, 1),test_class{2}(:, 2),'r+');
l1 = line([1.28, 1.28], [-5, 5], 'Color', 'g', 'LineWidth', 2);
legend(l1, 'x = 1.28')