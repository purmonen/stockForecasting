clear all;

stockPrices = csvread('omx30.csv');
stockPrices = stockPrices(end:-1:1);
stockPrices = stockPrices(1:300);

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

windowSize = 5;
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
trainingSize = 100;

if trainingSize > size(patterns,2) 
   trainingSize = size(patterns,2);
end


pricePatterns = [];
priceTargets = [];
for i = 1+windowSize:length(stockPrices)
    pricePatterns = [pricePatterns, stockPrices(i-windowSize:i-1)];
    priceTargets = [priceTargets, stockPrices(i)];
end

trnData = [pricePatterns(:,1:trainingSize)' priceTargets(:,1:trainingSize)'];
fis = anfis(trnData);
fis = anfis(trnData, fis, [10 0 0.01 0.9 1.1], [1 1 1 1], [pricePatterns' priceTargets']);

fis2 = anfis([patterns(:, 1:trainingSize)' targets(:, 1:trainingSize)']);
fis2 = anfis([patterns(:, 1:trainingSize)' targets(:, 1:trainingSize)'], fis, [10 0 0.01 0.9 1.1], [1 1 1 1], [patterns' targets']);

%trainingSize = size(patterns,2);armax
perceptron.validationPatterns = patterns(:, trainingSize+1:end);
perceptron.validationTargets = targets(:, trainingSize+1:end);
perceptron.train(patterns(:, 1:trainingSize), targets(1:trainingSize));
out = perceptron.recall(patterns);

%priceChanges / priceScalar - 0.5
predictedPrices = stockPrices(windowSize+1:end-1).*(out'*normalizedScalar+normalizedOffset);
predictedPricesAnfis = stockPrices(windowSize+1:end-1).*(evalfis(patterns,fis2)*normalizedScalar+normalizedOffset);
getPrediction = @(v) v(end-size(predictedPrices,1)+1:end);
meanSize = 1;
means = getPrediction(arrayfun(@(i) mean(stockPrices(i-meanSize:i-1)), meanSize+1:size(stockPrices,1)))';
leastSquares = getPrediction(arrayfun(@(i) 2*stockPrices(i-1) - stockPrices(i-2), 3:size(stockPrices,1)))';
stockPrices = getPrediction(stockPrices);
anfisPrices = getPrediction(evalfis(pricePatterns, fis));

getError = @(v) sum(abs(stockPrices - v) ./ stockPrices) / size(stockPrices,1);
getErrorValidation = @(v) sum(abs(stockPrices(end-trainingSize:end) - v(end-trainingSize:end)) ./ stockPrices(end-trainingSize:end)) / size(stockPrices(end-trainingSize:end),1);

figure(1)
plot([stockPrices predictedPricesAnfis])

predictor = predictedPrices;
indexes = 1:length(predictor)-1;
growingIndexes = @(p) indexes(p(indexes+1)' > p(indexes)');
calculateCash = @(predictedPrices) prod(stockPrices(growingIndexes(predictedPrices)+1) ./ stockPrices(growingIndexes(predictedPrices)));


priceLabels = {'best'; 'pred'; 'mean'; 'leas'; 'anfi'; 'anf2'};
prices = [stockPrices predictedPrices means leastSquares anfisPrices predictedPricesAnfis];
for i=1:length(priceLabels)
    disp([strcat(priceLabels{i}, ': Err | Cash')])
    disp([getError(prices(:,i)), calculateCash(prices(:,i)) getErrorValidation(prices(:,i))])
end

randomCash = stockPrices(end) / stockPrices(1)

title('Apple Inc')
ylabel('Price')
xlabel('Day')
axis tight
legend('Real', 'Predicted', 'Least squares', 'Training size')
set(gcf,'color','w')


%figure(2)
%plot((predictedPrices/stockPrices-1)*100)

%figure(2);
%perceptron.plotErrors()
%set(gcf,'color','w')