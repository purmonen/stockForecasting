clear all;

allStockPrices = csvread('omx30.csv');
allStockPrices = allStockPrices(end:-1:1);

index = 1;

winner = [0,0,0];

sampleSize = 50;
while (index+1)*sampleSize <= length(allStockPrices)
    stockPrices = allStockPrices(index*sampleSize:(index+1)*sampleSize);
    disp(sprintf('Stock prices %d', index));
    index = index + 1;
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
    
    windowSize = 4;
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
    perceptron.hiddenNodes = 8;
    perceptron.eta = 0.01;
    trainingSize = 40;
    validationSize = 5;
      
    if trainingSize > size(patterns,2)
        trainingSize = size(patterns,2);
    end
   
    
    trainingInput = patterns(:, 1:(trainingSize-validationSize));
    trainingOutput = targets(:, 1:(trainingSize-validationSize));
    
    validationInput = patterns(:, (trainingSize-validationSize+1):(trainingSize));
    validationOutput = targets(:, (trainingSize-validationSize+1):(trainingSize));
    
    testInput = patterns(:, (trainingSize+1):end);
    testOutput = targets(:, (trainingSize+1):end);
    
    fis2 = anfis([trainingInput' trainingOutput'], [], [], [0 0 0 0], [validationInput' validationOutput'], 0);
    
    perceptron.validationPatterns = validationInput;
    perceptron.validationTargets = validationOutput;
    perceptron.train(trainingInput, trainingOutput);
    
    recallInput = testInput;
    recallOutput = testOutput;
    
    predictedPrices = stockPrices(end-length(recallInput):end-1).*(perceptron.recall(recallInput)'*normalizedScalar+normalizedOffset);
    predictedPricesAnfis = stockPrices(end-length(recallInput):end-1).*(evalfis(recallInput,fis2)*normalizedScalar+normalizedOffset);
    getPrediction = @(v) v(end-size(predictedPrices,1)+1:end);
    meanSize = 1;
    means = getPrediction(arrayfun(@(i) mean(stockPrices(i-meanSize:i-1)), meanSize+1:size(stockPrices,1)))';
    leastSquares = getPrediction(arrayfun(@(i) 2*stockPrices(i-1) - stockPrices(i-2), 3:size(stockPrices,1)))';
    stockPrices = getPrediction(stockPrices);
    
    meanPercentageError = @(v) sum(abs(stockPrices - v) ./ stockPrices) / size(stockPrices,1)*100;
    rootMeanSquareError = @(v) (sum((stockPrices - v).^2) / size(stockPrices,1))^0.5;
    getError2 = @(v) sum(abs(stockPrices - v)) / size(stockPrices,1);
    getErrorValidation = @(v) sum(abs(stockPrices(end-trainingSize:end) - v(end-trainingSize:end)) ...
        ./ stockPrices(end-trainingSize:end)) / size(stockPrices(end-trainingSize:end),1)*100;
    
    figure(1)
    plot([stockPrices predictedPricesAnfis])
    
    predictor = predictedPrices;
    indexes = 1:length(predictor)-1;
    growingIndexes = @(p) indexes(p(indexes+1)' > p(indexes)');
    calculateCash = @(predictedPrices) prod(stockPrices(growingIndexes(predictedPrices)+1) ./ stockPrices(growingIndexes(predictedPrices)));
    
    priceLabels = {'pred'; 'mean'; 'anfi'};
    prices = [predictedPrices means predictedPricesAnfis];
    
    smallestErr = inf;
    bestMethod = -1;
    for i=1:length(priceLabels)
        err = meanPercentageError(prices(:,i));
        if err < smallestErr
            smallestErr = err;
            bestMethod = i;
        end
        disp([strcat(priceLabels{i}, ': Cash')]);
        %disp([meanPercentageError(prices(:,i)) getError2(prices(:,i)) rootMeanSquareError(prices(:,i))])
        disp([calculateCash(prices(:,i))])
    end
    
    winner(bestMethod) = winner(bestMethod) + 1;
    
    randomCash = stockPrices(end) / stockPrices(1)
    
    title('OMX Stockholm 30')
    ylabel('Price')
    xlabel('Day')
    axis tight
    %legend('Real', 'Predicted', 'Least squares', 'Training size')
    set(gcf,'color','w')
end

winner