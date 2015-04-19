clear all;

% Open	High	Low	Close	Volume
allPrices = csvread('sp.csv');
allPrices = allPrices(1:end/80+300, :);
% Move closing price to first index so we won't need to change price index
allPrices = [allPrices(:, 4), allPrices(:, 1:3), allPrices(:, 5:end)];

% Reverse prices so they are in correct order
allPrices = allPrices(end:-1:1, :);
allPriceChanges = allPrices(2:end,:) ./ allPrices(1:end-1,:);

priceIndex = 1;
winner = [0,0,0,0, 0];
totalWinnings = [0,0,0,0,0];
winnerDuringDecline = [0,0,0,0,0];
totalWinningsDuringDecline = [0,0,0,0,0];
%sampleSize = floor(length(allPrices) / 10)
sampleSize = 12

%sampleSize = 300;
windowSize = 1;
trainingSize = sampleSize-windowSize-1
validationSize = 0;
numDecliningPeriods = 0;

allPredictedChanges = [];
allPredictedAnfisChanges = [];
for index=1:size(allPrices,1)-windowSize-sampleSize-2
    index
    priceChanges = allPriceChanges(index:sampleSize+index-1, :);
    a = allPrices(2:end,:);
    prices = a(index:sampleSize+index-1, :);
    
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
    perceptron.iterations = 100;
    perceptron.hiddenNodes = 10;
    perceptron.eta = 0.01;
    
    trainingInput = patterns(:, 1:(trainingSize-validationSize));
    trainingOutput = targets(:, 1:(trainingSize-validationSize));
    
    validationInput = patterns(:, (trainingSize-validationSize+1):(trainingSize));
    validationOutput = targets(:, (trainingSize-validationSize+1):(trainingSize));
    
    testInput = patterns(:, (trainingSize+1):end);
    testOutput = targets(:, (trainingSize+1):end);
    
    
    %fis = anfis([trainingInput' trainingOutput'], [], [], [0 0 0 0], [], 1);
    fis = genfis1([trainingInput' trainingOutput'])
    
    perceptron.validationPatterns = validationInput;
    perceptron.validationTargets = validationOutput;
    perceptron.train(trainingInput, trainingOutput);
    %predictedChanges = perceptron.recall(testInput)' * normalizedScalar(priceIndex) + normalizedOffset(priceIndex);
    allPredictedChanges = [allPredictedChanges; perceptron.recall(testInput)' * normalizedScalar(priceIndex) + normalizedOffset(priceIndex)];
    allPredictedAnfisChanges = [allPredictedAnfisChanges; evalfis(testInput,fis)' * normalizedScalar(priceIndex) + normalizedOffset(priceIndex)];
end
allBeforeRealPrices = allPrices(end-length(allPredictedChanges):end-1, priceIndex);
allRealPrices = allPrices(end-length(allPredictedChanges)+1:end, priceIndex);
allPredictedPrices = allPredictedChanges .* allBeforeRealPrices;
allPredictedAnfisPrices = allPredictedAnfisChanges .* allBeforeRealPrices;

plot([allRealPrices allPredictedPrices allPredictedAnfisPrices])
legend('Real', 'MLP', 'ANFIS')

meanPercentageError = @(v) sum(abs(allRealPrices - v) ./ allRealPrices) / length(allRealPrices)*100;


winner = [0,0,0,0];
totalWinnings = [0,0,0,0];
winnerDuringDecline = [0,0,0,0];
totalWinningsDuringDecline = [0,0,0,0];

periodSize = 30;
periodPredictedPrices = reshape(allPredictedPrices(1:length(allPredictedPrices)-mod(length(allPredictedPrices),periodSize)), periodSize, floor(length(allPredictedPrices) / periodSize));
periodPrices = reshape(allRealPrices(1:length(allRealPrices)-mod(length(allRealPrices),periodSize)), periodSize, floor(length(allRealPrices) / periodSize));
periodAnfisPrices = reshape(allPredictedAnfisPrices(1:length(allPredictedAnfisPrices)-mod(length(allPredictedAnfisPrices),periodSize)), periodSize, floor(length(allPredictedAnfisPrices) / periodSize));


for period=1:size(periodPredictedPrices,2)
    realPrices = periodPrices(:, period);
    predictedPrices = periodPredictedPrices(:, period);
    predictedAnfisPrices = periodAnfisPrices(:, period);
    prices = [predictedPrices predictedAnfisPrices predictedAnfisPrices];
    bestMethods = [4];
    indexes = 1:length(realPrices)-1;
    growingIndexes = @(p) indexes(p(indexes+1)' > p(indexes)');
    calculateCash = @(predictedPrices) prod(realPrices(growingIndexes(predictedPrices)+1) ./ realPrices(growingIndexes(predictedPrices)));
    
    randomCash = realPrices(end) / realPrices(1);
    greatestCash = randomCash;
    totalWinnings(4) = totalWinnings(4) + randomCash;
    for i=1:size(prices,2)
        cash = calculateCash(prices(:, i));
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
end

winner
winnings = totalWinnings / size(periodPrices,2)
winnerDuringDecline
winningsDuringDecline = totalWinningsDuringDecline
mlpMpe = meanPercentageError(allPredictedPrices)
anfisMpe = meanPercentageError(allPredictedAnfisPrices)
naiveMpe = meanPercentageError(allBeforeRealPrices)

dirCor = @(p) sum(((allRealPrices(2:end) ./ allRealPrices(1:end-1) - 1) .* (p(2:end) ./ p(1:end-1) - 1) > 0)) / (length(allRealPrices)-1)


sum(((allRealPrices(2:end) ./ allBeforeRealPrices(2:end) - 1) .* (allPredictedPrices(2:end) ./ allPredictedPrices(1:end-1) - 1) > 0)) / (length(allRealPrices)-1)