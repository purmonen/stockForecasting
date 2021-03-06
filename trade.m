function [money, transactions] = trade(realPrices, predictedPrices, transactionFee, sellLimit)
predictedCash = 1;
isIn = true;
transactions = [];
for i=2:length(predictedPrices)
    trans = 0;
    if ~isIn
        trans = predictedCash * transactionFee * 1;
    end
    if predictedPrices(i) >= predictedPrices(i-1) + trans
        predictedCash = (predictedCash-trans) * realPrices(i) / realPrices(i-1);
        isIn = true;
    elseif isIn && predictedPrices(i) >= predictedPrices(i-1) * sellLimit
        predictedCash = predictedCash * realPrices(i) / realPrices(i-1);
    else
        isIn = false;
        transactions = [transactions; i];
    end
end
money = predictedCash;
end