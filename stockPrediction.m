clear all;
r1 = 0;
r2 = 3900;
c1 = 0;
c2 = 0;
stockPrices = csvread('aapl.csv', r1, c1, [r1 c1 r2 c2]);

priceScalar = max(stockPrices);
stockPrices = stockPrices(end:-1:1) / priceScalar - 0.5;

windowSize = 7;
patterns = [];
targets = [];
means = [];
for i = 1:size(stockPrices,1)-windowSize
    patterns = [patterns, stockPrices(i:i+windowSize-1)];
    targets = [targets, stockPrices(i+windowSize)];
    means = [means, mean(stockPrices(i:i+windowSize-1))];
end

perceptron = MultilayerPerceptron();
perceptron.plottingEnabled = false;
perceptron.iterations = 3000;
perceptron.hiddenNodes = 15;
perceptron.eta = 0.01;
trainingSize = 1000;
%trainingSize = size(patterns,2);
perceptron.validationPatterns = patterns(:, trainingSize+1:end);
perceptron.validationTargets = targets(:, trainingSize+1:end);
perceptron.train(patterns(:, 1:trainingSize), targets(1:trainingSize));

out = perceptron.recall(patterns);
targets = targets;
out = out;
means = [zeros(1, windowSize-1), means(1:end-windowSize+1)];
figure(1)
plot(1:size(targets,2), [targets;out;means], [trainingSize trainingSize], [min([targets,out]) max([targets,out])], 'r')
title('Apple Inc')
ylabel('Price')
xlabel('Day')
axis tight
legend('Real', 'Predicted', 'Means', 'Training size')
set(gcf,'color','w')

figure(2);
perceptron.plotErrors()
set(gcf,'color','w')