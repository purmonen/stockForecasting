clear all;
r1 = 0;
r2 = 800;
c1 = 0;
c2 = 0;
stockPrices = csvread('aapl.csv', r1, c1, [r1 c1 r2 c2]);

priceScalar = max(stockPrices);
stockPrices = stockPrices(end:-1:1) / priceScalar - 0.5;
stockChanges = stockPrices(1:1:end-1) ./ stockPrices(2:1:end);
prices = stockPrices(end:-1:1);
stockPrices = stockChanges;

annError = [];
meanError = [];

%for hiddenNodes = 1:10

windowSize = 1;

    
patterns = [];
targets = [];
means = [];
for i = 1:size(stockPrices,1)-windowSize
    patterns = [patterns, stockPrices(i:i+windowSize-1)];
    targets = [targets, stockPrices(i+windowSize)];
    means = [means, mean(stockPrices(i:i+windowSize-1))];
end



trainingSize = min(500, length(patterns));

perceptron = MultilayerPerceptron();
perceptron.plottingEnabled = false;
perceptron.iterations = 3000;
perceptron.hiddenNodes = 10;
perceptron.eta = 0.01;

%trainingSize = 500;
%trainingSize = size(patterns,2);
perceptron.validationPatterns = patterns(:, trainingSize+1:end);
perceptron.validationTargets = targets(:, trainingSize+1:end);
perceptron.train(patterns(:, 1:trainingSize), targets(1:trainingSize));
perceptron
out = perceptron.recall(patterns);
targets = targets;
out = out;
means = [zeros(1, windowSize-1), means(1:end-windowSize+1)];
figure(1)

annError = [annError, sum((out - targets) .^ 2)];
meanError = [meanError, sum((means - targets) .^ 2)];
%end
%figure(1);
%hold off;
%plot(1:size(annError,2),[annError; meanError]);
%legend('annError', 'meanError');

targets = targets .* (prices(windowSize:end-2)');
out = out .* (prices(windowSize:end-2)');
means = means .* (prices(windowSize:end-2)');
hold off
plot(1:size(targets,2), [targets;out;means], [trainingSize trainingSize], [min([targets,out]) max([targets,out])], 'r')
title('Apple Inc')
ylabel('Price')
xlabel('Day')
axis tight
legend('Real', 'Predicted', 'Means', 'Training size')
set(gcf,'color','w')
hold on
figure(2);
perceptron.plotErrors()
set(gcf,'color','w')