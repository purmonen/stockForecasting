clear all;

% Open	High	Low	Close	Volume
allPrices = csvread('sp500.csv');
allPrices = allPrices(1:end, :);
% Move closing price to first index so we won't need to change price index
allPrices = [allPrices(:, 4), allPrices(:, 1:3), allPrices(:, 5:end)];

% Reverse prices so they are in correct order
allPrices = allPrices(end:-1:1, :);
allPriceChanges = allPrices(2:end,:) ./ allPrices(1:end-1,:);

priceIndex = 1;
winner = [0,0,0,0, 0];
winnerNew = [0,0,0,0, 0];
totalWinnings = [1,1,1,1,1];
totalWinningsNew = [0,0,0,0,0];
winnerDuringDecline = [0,0,0,0,0];
totalWinningsDuringDecline = [0,0,0,0,0];
%sampleSize = floor(length(allPrices) / 10)
sampleSize = 30

%sampleSize = 300;
windowSize = 3;
trainingSize = sampleSize-windowSize-1
validationSize = 0;
numDecliningPeriods = 0;

allPredictedChanges = [];
allPredictedAnfisChanges = [];
for index=1:size(allPrices,1)-windowSize-sampleSize-2
    %index
    priceChanges = allPriceChanges(index:sampleSize+index-1, :);
    
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
    perceptron.iterations = 30;
    perceptron.hiddenNodes = 3;
    perceptron.eta = 0.05;
    
    trainingInput = patterns(:, 1:(trainingSize-validationSize));
    trainingOutput = targets(:, 1:(trainingSize-validationSize));
    
    validationInput = patterns(:, (trainingSize-validationSize+1):(trainingSize));
    validationOutput = targets(:, (trainingSize-validationSize+1):(trainingSize));
    
    testInput = patterns(:, (trainingSize+1):end);
    testOutput = targets(:, (trainingSize+1):end);
    
    
    %fis = anfis([trainingInput' trainingOutput'], [], [], [0 0 0 0], [], 1);
    %fis = genfis1([trainingInput' trainingOutput'])
    
    perceptron.validationPatterns = validationInput;
    perceptron.validationTargets = validationOutput;
    perceptron.train(trainingInput, trainingOutput);
    %predictedChanges = perceptron.recall(testInput)' * normalizedScalar(priceIndex) + normalizedOffset(priceIndex);
    allPredictedChanges = [allPredictedChanges; perceptron.recall(testInput)' * normalizedScalar(priceIndex) + normalizedOffset(priceIndex)];
    %allPredictedAnfisChanges = [allPredictedAnfisChanges; evalfis(testInput,fis)' * normalizedScalar(priceIndex) + normalizedOffset(priceIndex)];
end
allBeforeRealPrices = allPrices(end-length(allPredictedChanges):end-1, priceIndex);
allRealPrices = allPrices(end-length(allPredictedChanges)+1:end, priceIndex);
allPredictedPrices = allPredictedChanges .* allBeforeRealPrices;
%allPredictedAnfisPrices = allPredictedAnfisChanges .* allBeforeRealPrices;

plot([allRealPrices allPredictedPrices])
legend('Real', 'MLP', 'ANFIS')

meanAbsolutePercentageError = @(v) sum(abs(allRealPrices - v) ./ allRealPrices) / length(allRealPrices)*100;
meanAbsoluteError = @(v) sum(abs(allRealPrices - v)) / length(allRealPrices)*100;

winner = [0,0,0,0];
totalWinnings = [1,1,1,1];
winnerDuringDecline = [0,0,0,0];
totalWinningsDuringDecline = [0,0,0,0];

periodSize = 90;
periodPredictedPrices = reshape(allPredictedPrices(1:length(allPredictedPrices)-mod(length(allPredictedPrices),periodSize)), periodSize, floor(length(allPredictedPrices) / periodSize));
periodPrices = reshape(allRealPrices(1:length(allRealPrices)-mod(length(allRealPrices),periodSize)), periodSize, floor(length(allRealPrices) / periodSize));
%periodAnfisPrices = reshape(allPredictedAnfisPrices(1:length(allPredictedAnfisPrices)-mod(length(allPredictedAnfisPrices),periodSize)), periodSize, floor(length(allPredictedAnfisPrices) / periodSize));


for period=1:size(periodPredictedPrices,2)
    realPrices = periodPrices(:, period);
    predictedPrices = periodPredictedPrices(:, period);
    %predictedAnfisPrices = periodAnfisPrices(:, period);
    prices = [predictedPrices];
    bestMethods = [4];
    indexes = 1:length(realPrices)-1;
    growingIndexes = @(p) indexes(p(indexes+1)' > p(indexes)');
    calculateCash = @(predictedPrices) prod(realPrices(growingIndexes(predictedPrices)+1) ./ realPrices(growingIndexes(predictedPrices)));
    randomCash = realPrices(end) / realPrices(1);
    greatestCash = randomCash;
    totalWinnings(4) = totalWinnings(4) * randomCash;
    for i=1:size(prices,2)
        p = prices(:, i);
        money = trade(realPrices, predictedPrices, 0.12/100, 0.96);
        totalWinnings(i) = totalWinnings(i) * money;
        if randomCash < 1
            totalWinningsDuringDecline(i) = totalWinningsDuringDecline(i) + money;
        end
        if money > greatestCash
            greatestCash = money;
            bestMethods = [i];
        elseif money == greatestCash
            bestMethods = [bestMethods; i];
        end
    end
    winner(bestMethods) = winner(bestMethods) + 1;
    if randomCash < 1
        numDecliningPeriods = numDecliningPeriods + 1;
        winnerDuringDecline(bestMethods) = winnerDuringDecline(bestMethods) + 1;
        totalWinningsDuringDecline(4) = totalWinningsDuringDecline(4) + randomCash;
    end
end

winner
winnings = totalWinnings / size(periodPrices,2)
winnerDuringDecline
winningsDuringDecline = totalWinningsDuringDecline
mlpMpe = meanAbsolutePercentageError(allPredictedPrices)

%anfisMpe = meanAbsolutePercentageError(allPredictedAnfisPrices)
naiveMpe = meanAbsolutePercentageError(allBeforeRealPrices);

dirCor = @(p) sum(((allRealPrices(2:end) ./ allRealPrices(1:end-1) - 1) .* (p(2:end) ./ p(1:end-1) - 1) > 0)) / (length(allRealPrices)-1);
sum(((allRealPrices(2:end) ./ allBeforeRealPrices(2:end) - 1) .* (allPredictedPrices(2:end) ./ allPredictedPrices(1:end-1) - 1) > 0)) / (length(allRealPrices)-1);


% New way of predicting cash - can add transaction cost!
predictedCash99 = trade(allRealPrices, allPredictedPrices, 0.12/100, 0.96)

% S&P: 1.5641, FTSE: 2.4900, HSI: 1.9061

naiveCash = allRealPrices(end) / allRealPrices(1)

x = 0.85:0.0001:1;
y = arrayfun(@(x) trade(allRealPrices, allPredictedPrices, 0.12/100, x), x);
plot(x, y, 'b')
hold on
plot([0.85 1], [naiveCash naiveCash],'g')

% Old way of predicting cash
predictedCash2 = 1;
for i=2:length(allPredictedPrices)
    if allPredictedPrices(i) >= allRealPrices(i-1)
        predictedCash2 = predictedCash2 * allRealPrices(i) / allRealPrices(i-1);
    end
end
predictedCash2

indexes = 1:length(allRealPrices)-1;
growingIndexes = @(p) indexes(p(indexes+1)' > p(indexes)');
calculateCash = @(predictedPrices) prod(allRealPrices(growingIndexes(predictedPrices)+1) ./ allRealPrices(growingIndexes(predictedPrices)));
oldCash = calculateCash(allPredictedPrices)


(1 - sum(allRealPrices ./ allBeforeRealPrices - 1 > 0) / length(allRealPrices))^5 * 100