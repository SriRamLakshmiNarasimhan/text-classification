function [J theta] = costFunctionFMINUNC(theta, X, y)
  
  m = size(X,1);
  gradDescend = zeros(size(theta));
  pred = sigmoid(X, theta); 
  J = (-1/m)*sum((y.*log(pred) + (1-y).*log(1-pred))(:));
  gradDescend = X'*(pred - y);  
  
  endfunction