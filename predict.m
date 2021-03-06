function [prob predValue] = predict(string, allTheta, vocabList, map)
  predValue = -1;
  input = strsplit(string, " ");
  len = size(input,2);
  n = size(vocabList, 1);
  featureVector = zeros(1, n);
  
  for i = 1:len
    for j = 1:n
      if(strcmp(input{i}, vocabList{j}) ==1)
          featureVector(1, j) = 1;
        endif
      endfor
    endfor

    featureVector = [1 featureVector];
    [prob classification] = max(sigmoid(featureVector, allTheta));

    fprintf('Predicted that "%s" corresponds to "%s" classification with probability %f.\n', ...
		string, map{classification}, prob);

endfunction