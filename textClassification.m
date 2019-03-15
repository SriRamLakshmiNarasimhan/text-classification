close all; clear; clc;
fprintf('\n******************************************************************************************************\n');
fprintf('Beginning text classification...\n\n');

[X vocabList m n] = generateFeatureMatrix("test-X.txt", "test-vocab.txt");
[map yMultiClass] = generateOutputVector("test-y.txt");
k = size(yMultiClass, 2);
allTheta = zeros(n, k);
alpha = 1;
iterations = 500;
costHistory = zeros(iterations, k+1);
costHistory(:, 1) = 1:1:iterations;

  for i = 1: k
    [cost costHistory(:, i+1) allTheta(:, i)] = costFunction(X, yMultiClass(:, i), allTheta(:, i), alpha, iterations);
    hold on;
    subplot(1, k, i) = plot(costHistory(:,1), costHistory(:,i+1), '.', 'markersize', 10);   
  endfor;
  title('Cost Function'); xlabel('No. of iterations'); ylabel('Cost'); legend(map);

fprintf('Review plot and press enter to continue.\n\n');
pause;

fprintf('Cost is %f\nTheta is\n', cost);
allTheta

text = "good evening";
fprintf('Beginning prediction for provided text "%s"...\n\n', text);
predict(text, allTheta, vocabList, map);
fprintf('To continue prediction, execute predict("input text", allTheta, vocabList, map) in command line.\n')
fprintf('\n******************************************************************************************************\n');