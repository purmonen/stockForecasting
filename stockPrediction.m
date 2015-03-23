clear all;
r1 = 0;
r2 = 2000;
c1 = 0;
c2 = 0;
stockPrices = csvread('aapl.csv', r1, c1, [r1 c1 r2 c2]);
stockPrices = stockPrices(end:-1:1);

priceChanges = stockPrices(2:end) ./ stockPrices(1:end-1);
normalizedOffset = (max(priceChanges) - min(priceChanges)) / 2 + min(priceChanges);
%normalizedOffset = 1;
normalizedPriceChanges = (priceChanges - normalizedOffset);
normalizedScalar = max(normalizedPriceChanges);
%normalizedScalar = 1;
normalizedPriceChanges = normalizedPriceChanges / normalizedScalar;

%normalizedScalar = 40;
%normalizedPriceChanges = priceChanges;
%normalizedPriceChanges = stockPrices;

windowSize = 7;
patterns = [];
targets = [];
means = [];

for i = 1:size(normalizedPriceChanges,1)-windowSize
    patterns = [patterns, normalizedPriceChanges(i:i+windowSize-1)];
    targets = [targets, normalizedPriceChanges(i+windowSize)];
    means = [means, mean(normalizedPriceChanges(i:i+windowSize-1))];
end

perceptron = MultilayerPerceptron();
perceptron.plottingEnabled = false;
perceptron.iterations = 1000;
perceptron.hiddenNodes = 10;
perceptron.eta = 0.01;
trainingSize = 2000;

if trainingSize > size(patterns,2) 
   trainingSize = size(patterns,2);
end
%trainingSize = size(patterns,2);
perceptron.validationPatterns = patterns(:, trainingSize+1:end);
perceptron.validationTargets = targets(:, trainingSize+1:end);
perceptron.train(patterns(:, 1:trainingSize), targets(1:trainingSize));
out = perceptron.recall(patterns);

%priceChanges / priceScalar - 0.5
predictedPrices = stockPrices(windowSize+1:end-1).*(out'*normalizedScalar+normalizedOffset);
getPrediction = @(v) v(end-size(predictedPrices,1)+1:end);
meanSize = 1;
means = getPrediction(arrayfun(@(i) mean(stockPrices(i-meanSize:i-1)), meanSize+1:size(stockPrices,1)))';
leastSquares = getPrediction(arrayfun(@(i) 2*stockPrices(i-1) - stockPrices(i-2), 3:size(stockPrices,1)))';
stockPrices = getPrediction(stockPrices)

getError = @(v) sum(abs(stockPrices - v) ./ stockPrices) / size(stockPrices,1);
getPrediction(means) - getPrediction(stockPrices);

predictedErr = getError(predictedPrices)
meanErr = getError(means)
leastSquaresErr = getError(leastSquares)

figure(1)
plot([stockPrices predictedPrices])
%plot([stockPrices(1:end-1) predictedPrices(2:end)])

predictor = predictedPrices;

indexes = 1:length(predictor)-1;
growingIndexes = @(p) indexes(p(indexes+1)' > p(indexes)');


calculateCash = @(predictedPrices) prod(stockPrices(growingIndexes(predictedPrices)+1) ./ stockPrices(growingIndexes(predictedPrices)));
predictedCash = calculateCash(predictedPrices)
meanCash = calculateCash(means)
leastSquaresCash = calculateCash(leastSquares)
bestCash = calculateCash(stockPrices)
randomCash = stockPrices(end) / stockPrices(1)

title('Apple Inc')
ylabel('Price')
xlabel('Day')
axis tight
legend('Real', 'Predicted', 'Least squares', 'Training size')
set(gcf,'color','w')

%figure(2);
%perceptron.plotErrors()
%set(gcf,'color','w')