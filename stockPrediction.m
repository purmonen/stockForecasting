clear all;

% Open	High	Low	Close	Volume
allStockPrices = csvread('sp.csv');
allStockPrices = allStockPrices(1:300, :);
% Move closing price to first index so we won't need to change price index
allStockPrices = [allStockPrices(:, 4), allStockPrices(:, 1:3), allStockPrices(:, 5:end)];

% Reverse prices so they are in correct order
allStockPrices = allStockPrices(end:-1:1, :);
%allStockPrices = [allStockPrices(:, 1), allStockPrices(:, 5:5)];
%allStockPrices = allStockPrices(:, 1:5);
%allStockPrices = allStockPrices(:, 1);

priceIndex = 1;

%allStockPrices = allStockPrices(1:50);
%allStockPrices = (1:200)'.^3;

index = 0;
winner = [0,0,0,0];
totalWinnings = [0,0,0,0];

winnerDuringDecline = [0,0,0,0];
totalWinningsDuringDecline = [0,0,0,0];


sampleSize = floor(length(allStockPrices) / 1)

%sampleSize = 300;
windowSize = 4;
trainingSize = 170;
validationSize = 0;

numDecliningPeriods = 0;
while (index+1)*sampleSize <= length(allStockPrices)
    stockPrices = allStockPrices(index*sampleSize+1:(index+1)*sampleSize, :);
    assert(length(stockPrices) == sampleSize)
    
    disp(sprintf('Stock prices %d', index));
    index = index + 1;
    priceChanges = stockPrices(2:end,:) ./ stockPrices(1:end-1,:);
    assert(sum(sum(abs(priceChanges .* stockPrices(1:end-1,:) - stockPrices(2:end,:)))) < 0.0001)
    
    normalizedOffset = (max(priceChanges) - min(priceChanges)) / 2 + min(priceChanges);
    normalizedPriceChanges = priceChanges;
    for i = 1:size(priceChanges,2)
        normalizedPriceChanges(:, i) = normalizedPriceChanges(:, i) - normalizedOffset(i);
    end
    normalizedScalar = max(normalizedPriceChanges);
    for i = 1:size(priceChanges,2)
        normalizedPriceChanges(:, i) = normalizedPriceChanges(:, i) / normalizedScalar(i);
    end
    
    patterns = [];
    targets = [];
    means = [];
    
    for i = 1:size(normalizedPriceChanges,1)-windowSize
        windowPatterns = normalizedPriceChanges(i:i+windowSize-1, :);
        patterns = [patterns, reshape(windowPatterns, numel(windowPatterns),1)];
        
        windowTargets = normalizedPriceChanges(i+windowSize,:);
        targets = [targets, windowTargets(priceIndex)];
        
        windowMeans =  mean(normalizedPriceChanges(i:i+windowSize-1, :));
        means = [means, reshape(windowMeans, numel(windowMeans),1)];
    end
    
    perceptron = MultilayerPerceptron();
    perceptron.plottingEnabled = false;
    perceptron.iterations = 500;
    perceptron.hiddenNodes = 20;
    perceptron.eta = 0.01;
    
    trainingInput = patterns(:, 1:(trainingSize-validationSize));
    trainingOutput = targets(:, 1:(trainingSize-validationSize));
    
    validationInput = patterns(:, (trainingSize-validationSize+1):(trainingSize));
    validationOutput = targets(:, (trainingSize-validationSize+1):(trainingSize));
    
    testInput = patterns(:, (trainingSize+1):end);
    testOutput = targets(:, (trainingSize+1):end);
    
    % These are the prices we want to predict!
    realPrices = stockPrices(end-length(testOutput)+1:end, priceIndex);
    oneDayBeforeRealPrices = stockPrices(end-length(testOutput):end-1, priceIndex);
    
    %testInput = trainingInput;
    %testOutput = trainingOutput;
    
    %fis2 = anfis([trainingInput' trainingOutput'], [], [], [0 0 0 0], [validationInput' validationOutput'], 0);
    %fis2 = anfis([trainingInput' trainingOutput']);
    
    perceptron.validationPatterns = validationInput;
    perceptron.validationTargets = validationOutput;
    perceptron.train(trainingInput, trainingOutput);
    predictedChanges = perceptron.recall(testInput)' * normalizedScalar(priceIndex) + normalizedOffset(priceIndex);
    predictedPrices =  predictedChanges .* oneDayBeforeRealPrices;

    meanPercentageError = @(v) sum(abs(realPrices - v) ./ realPrices) / length(realPrices)*100;
    %predictedPricesAnfis = stockPrices(end-length(recallInput):end-1).*(evalfis(recallInput,fis2)*normalizedScalar+normalizedOffset);    
    rootMeanSquareError = @(v) (sum((realPrices - v).^2) / length(realPrices))^0.5;
    indexes = 1:length(predictedPrices)-1;
    growingIndexes = @(p) indexes(p(indexes+1)' > p(indexes)');
    calculateCash = @(predictedPrices) prod(realPrices(growingIndexes(predictedPrices)+1) ./ realPrices(growingIndexes(predictedPrices)));
    
    prices = [predictedPrices];
    randomCash = realPrices(end) / realPrices(1);
    greatestCash = randomCash;
    totalWinnings(4) = totalWinnings(4) + randomCash;
    bestMethods = [4];
   
    for i=1:size(prices,2)
        cash = calculateCash(prices(:,i));
        totalWinnings(i) = totalWinnings(i) + cash;
        if randomCash < 1
            totalWinningsDuringDecline(i) = totalWinningsDuringDecline(i) + cash;
        end
        if cash > greatestCash
            greatestCash = cash;
            bestMethods = [i];
        elseif cash == greatestCash
            bestMethods = [bestMethods; i];
        end
    end
    winner(bestMethods) = winner(bestMethods) + 1;
    if randomCash < 1
        numDecliningPeriods = numDecliningPeriods + 1;
        winnerDuringDecline(bestMethods) = winnerDuringDecline(bestMethods) + 1;
        totalWinningsDuringDecline(4) = totalWinningsDuringDecline(4) + randomCash;
    end
    mpe = meanPercentageError(predictedPrices)
    title('OMX Stockholm 30')
    ylabel('Price')
    xlabel('Day')
    axis tight
    %legend('Real', 'Predicted', 'Least squares', 'Training size')
    set(gcf,'color','w')
    figure(1)
    plot([realPrices predictedPrices])
    legend('realPrices', 'predictedPrices')
end

winner
winnings = totalWinnings / index
winnerDuringDecline
winningsDuringDecline = totalWinningsDuringDecline / numDecliningPeriods


