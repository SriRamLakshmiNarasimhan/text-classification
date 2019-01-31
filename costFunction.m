function [cost costHistory theta] = costFunction(X, y, theta, alpha, iterations)    costHistory = zeros(iterations, 1);    for i=1:iterations    m = size(X,1);    pred = sigmoid(X, theta);     gradDescend = X'*(pred - y);      theta = theta - (alpha/m)*gradDescend;    cost = (-1/m)*sum((y.*log(pred) + (1-y).*log(1-pred))(:));    costHistory(i) = cost;  endfor    
endfunction
