close all; clear; clc;

[X vocabList m n] = generateFeatureMatrix("X.txt", "vocab.txt");
[y map] = generateOutputVector("y.txt");
theta = zeros(n, 1);


% gradient descend
alpha = 0.3;
iterations = 5000;
[J theta] = costFunction(X, y, theta, alpha, iterations);


% gradient descend using fminunc
%options = optimset('GradObj', 'on', 'MaxIter', 400);
%[J theta] = fminunc(@(t)(costFunctionFMINUNC(t, X, y)), theta, options);

fprintf('Cost is %f\nTheta is\n', J);
fprintf('%f\n', theta);

text = "hello there";
predict(text, theta, vocabList, map);

fprintf('\n******************************************************************************************************\n')


