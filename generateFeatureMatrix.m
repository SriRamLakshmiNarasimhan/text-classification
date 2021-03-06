function [featureMatrix vocabList m n] = generateFeatureMatrix(inputfile, vocabfile)
  
  m = readFileLength(inputfile);
  n = readFileLength(vocabfile);
  featureMatrix = zeros(m, n);
  
  data = getStringCell(inputfile);
  vocabList = getStringCell(vocabfile);
  
  for i = 1:m
    input = strsplit(data{i}, " ");
    for j = 1:size(input, 2)
      for k = 1:n     
        if(strcmp(input{j}, vocabList{k}) ==1)
          featureMatrix(i, k) =1;
        endif
      endfor
    endfor   
  endfor
  featureMatrix = [ones(m, 1) featureMatrix];
  n = size(featureMatrix, 2);
	
endfunction