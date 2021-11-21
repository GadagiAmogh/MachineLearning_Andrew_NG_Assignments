%% Machine Learning: Programming Exercise 5
%% Regularized Linear Regression and Bias vs. Variance
% In this exercise, you will implement regularized linear regression and use 
% it to study models with different bias-variance properties.
% Files needed for this exercise
%% 
% * |ex5.mlx| - MATLAB Live Script that steps you through the exercise
% * |ex5data1.mat| - Dataset
% * |submit.m| - Submission script that sends your solutions to our servers
% * |featureNormalize.m| - Feature normalization function
% * |fmincg.m| - Function minimization routine (similar to |fminunc|)
% * |plotFit.m| - Plot a polynomial fit
% * |trainLinearReg.m| - Trains linear regression using your cost function
% * *|linearRegCostFunction.m| - Regularized linear regression cost function
% * |*learningCurve.m| - Generates a learning curve
% * *|polyFeatures.m| - Maps data into polynomial feature space
% * *|validationCurve.m| - Generates a cross validation curve
%% 
% _**indicates files you will need to complete*_
% Clear existing variables and confirm that your Current Folder is set correctly
% Click into this section, then click the 'Run Section' button above. This will 
% execute the |clear| command to clear existing variables and the |dir| command 
% to list the files in your Current Folder. The output should contain all of the 
% files listed above and the 'lib' folder. If it does not, right-click the 'ex5' 
% folder and select 'Open' before proceding or see the instructions in |README.mlx| 
% for more details.

clear
dir
% Before you begin
% The workflow for completing and submitting the programming exercises in MATLAB 
% Online differs from the original course instructions. Before beginning this 
% exercise, make sure you have read through the instructions in |README.mlx| which 
% is included with the programming exercise files. |README| also contains solutions 
% to the many common issues you may encounter while completing and submitting 
% the exercises in MATLAB Online. Make sure you are following instructions in 
% |README| and have checked for an existing solution before seeking help on the 
% discussion forums.
%% 1. Regularized Linear Regression
% In the first half of the exercise, you will implement regularized linear regression 
% to predict the amount of water flowing out of a dam using the change of water 
% level in a reservoir. In the next half, you will go through some diagnostics 
% of debugging learning algorithms and examine the effects of bias vs. variance.
% 1.1 Visualizing the dataset
% We will begin by visualizing the dataset containing historical records on 
% the change in the water level, $x$, and the amount of water flowing out of the 
% dam, $y$. This dataset is divided into three parts:
%% 
% * A *training* set that your model will learn on: |X, y|
% * A *cross validation* set for determining the regularization parameter: |Xval, 
% yval|
% * A *test* set for evaluating performance. These are 'unseen' examples which 
% your model did not see during training: |Xtest, ytest|
%% 
% The code below will plot the training data (Figure 1). In the following parts, 
% you will implement linear regression and use that to fit a straight line to 
% the data and plot learning curves. Following that, you will implement polynomial 
% regression to find a better fit to the data.
% 
% 

% Load from ex5data1: 
% You will have X, y, Xval, yval, Xtest, ytest in your environment
load ('ex5data1.mat');
% m = Number of examples
m = size(X, 1);

% Plot training data
figure;
plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
xlabel('Change in water level (x)');
ylabel('Water flowing out of the dam (y)');
% 1.2 Regularized linear regression cost function
% Recall that regularized linear regression has the following cost function:
% 
% $$J\left(\theta \right)=\frac{1}{2m}\left(\sum_{i=1}^m {\left(h_{\theta \;} 
% \left(x^{\left(i\right)} \right)-y^{\left(i\right)} \right)}^2 \right)+\frac{\;\lambda 
% }{2m}\left(\sum_{j=1}^n \theta_j^2 \right)$$
% 
% where $\lambda$ is a regularization parameter which controls the degree of 
% regularization (thus, helps preventing overfitting). The regularization term 
% puts a penalty on the overall cost $J$. As the magnitudes of the model parameters 
% $\theta_j$ increase, the penalty increases as well. Note that you should not 
% regularize the $\theta_0$ term. (In MATLAB, the $\theta_0$ term is represented 
% as |theta(1)| since indexing in MATLAB starts from 1).
% 
% You should now complete the code in the file |linearRegCostFunction.m|. Your 
% task is to write a function to calculate the regularized linear regression cost 
% function. If possible, try to vectorize your code and avoid writing loops. When 
% you are finished, the code below will run your cost function using |theta| initialized 
% at |[1; 1]|. You should expect to see an output of 303.993.

theta = [1 ; 1];
J = linearRegCostFunction([ones(m, 1) X], y, theta, 1);
fprintf('Cost at theta = [1 ; 1]: %f', J);
%% 
% _You should now submit your solutions.  Enter_ |*submit*| _at the command 
% prompt, then enter or confirm your login and token when prompted._
% 1.3 Regularized linear regression gradient
% Correspondingly, the partial derivative of regularized linear regression's 
% cost for $\theta_j$ is defined as
% 
% $$\begin{array}{l}\frac{\partial J\left(\theta \right)}{\partial \theta_0 
% }=\frac{\;1}{m}\sum_{i=1}^m \;\left(h_{\theta \;} \left(x^{\left(i\right)} \right)-y^{\left(i\right)} 
% \right)x^{\left(i\right)} \;\;\mathrm{for}\;\;j=0\\\frac{\partial J\left(\theta 
% \right)}{\partial \theta_j }=\left(\frac{\;1}{m}\sum_{i=1}^m \;\left(h_{\theta 
% \;} \left(x^{\left(i\right)} \right)-y^{\left(i\right)} \right)x_j^{\left(i\right)} 
% \right)+\frac{\lambda }{m}\theta_j \;\;\;\mathrm{for}\;\;j>0\end{array}$$
% 
% In |linearRegCostFunction.m|, add code to calculate the gradient, returning 
% it in the variable |grad|. When you are finished, the code below will run your 
% gradient function using theta initialized at [1; 1]. You should expect to see 
% a gradient of [-15.30; 598.250].

[J, grad] = linearRegCostFunction([ones(m, 1) X], y, theta, 1);
fprintf('Gradient at theta = [1 ; 1]:  [%f; %f] \n',grad(1), grad(2));
%% 
% _You should now submit your solutions.  Enter_ |*submit*| _at the command 
% prompt, then enter or confirm your login and token when prompted._
% 1.4 Fitting linear regression
% Once your cost function and gradient are working correctly, the code in this 
% section will run the code in |trainLinearReg.m| to compute the optimal values 
% of $\theta$. This training function uses |fmincg| to optimize the cost function. 
% In this part, we set regularization parameter $\lambda$ to zero. Because our 
% current implementation of linear regression is trying to fit a 2-dimensional 
% $\theta$, regularization will not be incredibly helpful for a $\theta$ of such 
% low dimension. In the later parts of the exercise, you will be using polynomial 
% regression with regularization.

%  Train linear regression with lambda = 0
lambda = 0;
[theta] = trainLinearReg([ones(m, 1) X], y, lambda);
%% 
% Finally, the code below should also plot the best fit line, resulting in an 
% image similar to Figure 2. The best fit line tells us that the model is not 
% a good fit to the data because the data has a nonlinear pattern. 

%  Plot fit over the data
figure;
plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
xlabel('Change in water level (x)');
ylabel('Water flowing out of the dam (y)');
hold on;
plot(X, [ones(m, 1) X]*theta, '--', 'LineWidth', 2)
hold off;
%% 
% 
% 
% While visualizing the best fit as shown is one possible way to debug your 
% learning algorithm, it is not always easy to visualize the data and model. In 
% the next section, you will implement a function to generate learning curves 
% that can help you debug your learning algorithm even if it is not easy to visualize 
% the data.
%% 2. Bias-variance
% An important concept in machine learning is the bias-variance tradeoff. Models 
% with high bias are not complex enough for the data and tend to underfit, while 
% models with high variance overfit the training data. In this part of the exercise, 
% you will plot training and test errors on a learning curve to diagnose bias-variance 
% problems.
% 2.1 Learning curves
% You will now implement code to generate the learning curves that will be useful 
% in debugging learning algorithms. Recall that a learning curve plots training 
% and cross validation error as a function of training set size. Your job is to 
% fill in |learningCurve.m| so that it returns a vector of errors for the training 
% set and cross validation set.
% 
% To plot the learning curve, we need a training and cross validation set error 
% for different training set sizes. To obtain different training set sizes, you 
% should use different subsets of the original training set |X|. Specically, for 
% a training set size of |i|, you should use the first |i| examples (i.e., |X(1:i,:)| 
% and |y(1:i))|. You can use the |trainLinearReg| function to find the $\theta$ 
% parameters. Note that |lambda| is passed as a parameter to the |learningCurve| 
% function. After learning the $\theta$ parameters, you should compute the error 
% on the training and cross validation sets. Recall that the training error for 
% a dataset is defined as
% 
% $$J_{\textrm{train}} \left(\theta \right)=\frac{1}{2m}\left\lbrack \sum_{i=1}^m 
% {\left(h_{\theta \;} \left(x^{\left(i\right)} \right)-y^{\left(i\right)} \right)}^2 
% \right\rbrack$$
% 
% In particular, note that the training error does not include the regularization 
% term. One way to compute the training error is to use your existing cost function 
% and set $\lambda$ to 0 only when using it to compute the training error and 
% cross validation error. When you are computing the training set error, make 
% sure you compute it on the training subset (i.e., |X(1:n,:)| and |y(1:n)|, instead 
% of the entire training set). However, for the cross validation error, you should 
% compute it over the entire cross validation set. You should store the computed 
% errors in the vectors |error_train| and |error_val|.  
% 
% In Figure 3, you can observe that both the train error and cross validation 
% error are high when the number of training examples is increased. This reflects 
% a high bias problem in the model - the linear regression model is too simple 
% and is unable to fit our dataset well. In the next section, you will implement 
% polynomial regression to fit a better model for this dataset.
% 
% 
% 
% When you are finished, run the code below to compute the learning curves and 
% produce a plot similar to Figure 3.

lambda = 0;
[error_train, error_val] = learningCurve([ones(m, 1) X], y, [ones(size(Xval, 1), 1) Xval], yval, lambda);

plot(1:m, error_train, 1:m, error_val);
title('Learning curve for linear regression')
legend('Train', 'Cross Validation')
xlabel('Number of training examples')
ylabel('Error')
axis([0 13 0 150])

fprintf('# Training Examples\tTrain Error\tCross Validation Error\n');
for i = 1:m
    fprintf('  \t%d\t\t%f\t%f\n', i, error_train(i), error_val(i));
end
%% 3. Polynomial regression
% The problem with our linear model was that it was too simple for the data 
% and resulted in underfitting (high bias). In this part of the exercise, you 
% will address this problem by adding more features. For use polynomial regression, 
% our hypothesis has the form:
% 
% $$h_\theta(x) =\theta_0+\theta_1\ast (waterLevel)+\theta_2*(waterLevel)^2+\cdots+\theta_p\ast 
% (waterLevel)^p \\\qquad\;=\theta_0+\theta_1 x_1+\theta_2 x_2+\cdots+\theta_p 
% x_p \end{aligned}$$
% 
% Notice that by defining $x_1 = (\text{waterLevel}), x_2 = (\text{waterLevel)}^2,\ldots,x_p 
% = (\text{waterLevel})^p$, we obtain a linear regression model where the features 
% are the various powers of the original value (_waterLevel_).
% 
% Now, you will add more features using the higher powers of the existing feature 
% $x$ in the dataset. Your task in this part is to complete the code in |polyFeatures.|m 
% so that the function maps the original training set |X| of size $m\times 1$into 
% its higher powers. Specifically, when a training set $X$ of size $m\times 1$ 
% is passed into the function, the function should return a $m\times p$ matrix 
% |X_poly|, where column 1 holds the original values of |X|, column 2 holds the 
% values of |X.^2|, column 3 holds the values of |X.^3|, and so on. Note that 
% you don't have to account for the zero-th power in this function. Now that you 
% have a function that will map features to a higher dimension, the code in the 
% next section will apply it to the training set, the test set, and the cross 
% validation set (which you haven't used yet).
%% 
% _You should now submit your solutions.  Enter_ |*submit*| _at the command 
% prompt, then enter or confirm your login and token when prompted._
% 3.1 Learning Polynomial Regression
% After you have completed |polyFeatures.m|, run the code below to train polynomial 
% regression using your linear regression cost function. Keep in mind that even 
% though we have polynomial terms in our feature vector, we are still solving 
% a linear regression optimization problem. The polynomial terms have simply turned 
% into features that we can use for linear regression. We are using the same cost 
% function and gradient that you wrote for the earlier part of this exercise.
% 
% For this part of the exercise, you will be using a polynomial of degree 8. 
% It turns out that if we run the training directly on the projected data, it 
% will not work well as the features would be badly scaled (e.g., an example with 
% $x = 40$ will now have a feature $x_8 = 40^8 = 6:5\times 10^{12$) Therefore, 
% you will need to use feature normalization. Before learning the parameters $\theta$ 
% for the polynomial regression, code in below will first call |featureNormalize| 
% to normalize the features of the training set, storing the |mu|, |sigma| parameters 
% separately. We have already implemented this function for you and it is the 
% same function from the first exercise.

p = 8;

% Map X onto Polynomial Features and Normalize
X_poly = polyFeatures(X, p);
[X_poly, mu, sigma] = featureNormalize(X_poly);  % Normalize
X_poly = [ones(m, 1), X_poly];                   % Add Ones

% Map X_poly_test and normalize (using mu and sigma)
X_poly_test = polyFeatures(Xtest, p);
X_poly_test = X_poly_test-mu; % uses implicit expansion instead of bsxfun
X_poly_test = X_poly_test./sigma; % uses implicit expansion instead of bsxfun
X_poly_test = [ones(size(X_poly_test, 1), 1), X_poly_test];         % Add Ones

% Map X_poly_val and normalize (using mu and sigma)
X_poly_val = polyFeatures(Xval, p);
X_poly_val = X_poly_val-mu; % uses implicit expansion instead of bsxfun
X_poly_val = X_poly_val./sigma; % uses implicit expansion instead of bsxfun
X_poly_val = [ones(size(X_poly_val, 1), 1), X_poly_val];           % Add Ones

fprintf('Normalized Training Example 1:\n');
fprintf('  %f  \n', X_poly(1, :));
%%
% Train the model
lambda = 0;
[theta] = trainLinearReg(X_poly, y, lambda);
%% 
% After learning the parameters $\theta$, the code below will generate two plots 
% (Figures 4,5) for polynomial regression with $\lambda = 0$. 
% 
% 
% 
% From Figure 4, you should see that the polynomial fit is able to follow the 
% datapoints very well - thus, obtaining a low training error. However, the polynomial 
% fit is very complex and even drops off at the extremes. This is an indicator 
% that the polynomial regression model is overfitting the training data and will 
% not generalize well.

% Plot training data and fit
plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
plotFit(min(X), max(X), mu, sigma, theta, p);
xlabel('Change in water level (x)');
ylabel('Water flowing out of the dam (y)');
title (sprintf('Polynomial Regression Fit (lambda = %f)', lambda));
[error_train, error_val] = learningCurve(X_poly, y, X_poly_val, yval, lambda);
plot(1:m, error_train, 1:m, error_val);
title(sprintf('Polynomial Regression Learning Curve (lambda = %f)', lambda));
xlabel('Number of training examples')
ylabel('Error')
axis([0 13 0 100])
legend('Train', 'Cross Validation')
%% 
% To better understand the problems with the unregularized $\lambda=0$ model, 
% you can see that the learning curve (Figure 5) shows the same effect where the 
% low training error is low, but the cross validation error is high. There is 
% a gap between the training and cross validation errors, indicating a high variance 
% problem. One way to combat the overfitting (high-variance) problem is to add 
% regularization to the model. In the next section, you will get to try different 
% $\lambda$ parameters to see how regularization can lead to a better model.
% 3.2 Optional (ungraded) exercise: Adjusting the regularization parameter
% In this section, you will get to observe how the regularization parameter 
% affects the bias-variance of regularized polynomial regression. You should now 
% modify the the |lambda| parameter in the code below and try $\lambda = 1,\; 
% 100$. 

% Choose the value of lambda
lambda = 100;
[theta] = trainLinearReg(X_poly, y, lambda);

% Plot training data and fit
plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
plotFit(min(X), max(X), mu, sigma, theta, p);
xlabel('Change in water level (x)');
ylabel('Water flowing out of the dam (y)');
title (sprintf('Polynomial Regression Fit (lambda = %f)', lambda));
[error_train, error_val] = learningCurve(X_poly, y, X_poly_val, yval, lambda);
plot(1:m, error_train, 1:m, error_val);
title(sprintf('Polynomial Regression Learning Curve (lambda = %f)', lambda));
xlabel('Number of training examples')
ylabel('Error')
axis([0 13 0 100])
legend('Train', 'Cross Validation')
%% 
% For each of these values, the code should generate a polynomial fit to the 
% data and also a learning curve.
% 
% 
% 
% 
% 
% For $\lambda= 1$, you should see a polynomial fit that follows the data trend 
% well (Figure 6) and a learning curve (Figure 7) showing that both the cross 
% validation and training error converge to a relatively low value. This shows 
% the $\lambda= 1$ regularized polynomial regression model does not have the high 
% bias or high-variance problems. In effect, it achieves a good trade-off between 
% bias and variance.
% 
% For $\lambda= 100$, you should see a polynomial fit (Figure 8) that does not 
% follow the data well. In this case, there is too much regularization and the 
% model is unable to fit the training data.
% 
% 
% 3.3 Selecting lambda using a cross validation set
% From the previous parts of the exercise, you observed that the value of $\lambda$ 
% can significantly affect the results of regularized polynomial regression on 
% the training and cross validation set. In particular, a model without regularization 
% ($\lambda = 0$) fits the training set well, but does not generalize. Conversely, 
% a model with too much regularization ($\lambda=100$)  does not fit the training 
% set and testing set well. A good choice of $\lambda$ (e.g. $\lambda=1$) can 
% provide a good fit to the data.
% 
% In this section, you will implement an automated method to select the parameter. 
% Concretely, you will use a cross validation set to evaluate how good each $\lambda$ 
% value is. After selecting the best $\lambda$ value using the cross validation 
% set, we can then evaluate the model on the test set to estimate how well the 
% model will perform on actual unseen data. Your task is to complete the code 
% in |validationCurve.m|. Specifically, you should should use the |trainLinearReg| 
% function to train the model using different values of $\lambda$ and compute 
% the training error and cross validation error. The function will try $\lambda$ 
% in the following range: {0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10}.
% 
% After you have completed the code, the code below will run your function and 
% plot a cross validation curve of error v.s $\lambda$ that allows you select 
% which $\lambda$ parameter to use. You should see a plot similar to Figure 9. 

[lambda_vec, error_train, error_val] = validationCurve(X_poly, y, X_poly_val, yval);
plot(lambda_vec, error_train, lambda_vec, error_val);
legend('Train', 'Cross Validation');
xlabel('lambda');
ylabel('Error');
for i = 1:length(lambda_vec)
    if i == 1
        fprintf('lambda\t\tTrain Error\tValidation Error\n');
    end
    fprintf('%f\t%f\t%f\n',lambda_vec(i), error_train(i), error_val(i));
end
%% 
% 
% 
% In this figure, we can see that the best value of $\lambda$ is around 3. Due 
% to randomness in the training and validation splits of the dataset, the cross 
% validation error can sometimes be lower than the training error. 
%% 
% _You should now submit your solutions.  Enter_ |*submit*| _at the command 
% prompt, then enter or confirm your login and token when prompted._
% 3.4 Optional (ungraded) exercise: Computing test set error
% In the previous part of the exercise, you implemented code to compute the 
% cross validation error for various values of the regularization parameter $\lambda$. 
% However, to get a better indication of the model's performance in the real world, 
% it is important to evaluate the 'final' model on a test set that was not used 
% in any part of training (that is, it was neither used to select the $\lambda$ 
% parameters, nor to learn the model parameters $\theta$).
% 
% For this optional (ungraded) exercise, you should compute the test error using 
% the best value of $\lambda$ you found. In our cross validation, we obtained 
% a test error of 3.8599 for $\lambda= 3.$ You do not need to submit any solutions 
% for this optional (ungraded) exercise.

%%%%%%%%% Add your code to compute the test error below %%%%%%%%%%%
[min_err, idx] = min(error_val);
lambda = lambda_vec(idx);
[theta] = trainLinearReg(X_poly, y, lambda);
[J_test, grad] = linearRegCostFunction(X_poly_test, ytest, theta, 0);   % lambda = 0 when calculate training error
fprintf('lambda=%f, test_error = %f\n', lambda, J_test);

fprintf('Program paused. Press enter to continue.\n');



% 3.5 Optional (ungraded) exercise: Plotting learning curves with randomly selected examples
% In practice, especially for small training sets, when you plot learning curves 
% to debug your algorithms, it is often helpful to average across multiple sets 
% of randomly selected examples to determine the training error and cross validation 
% error. Concretely, to determine the training error and cross validation error 
% for $i$ examples, you should first randomly select $i$ examples from the training 
% set and $i$ examples from the cross validation set. You will then learn the 
% parameters $\theta$ using the randomly chosen training set and evaluate the 
% parameters $\theta$ on the randomly chosen training set and cross validation 
% set. The above steps should then be repeated multiple times (say 50) and the 
% averaged error should be used to determine the training error and cross validation 
% error for $i$ examples.
% 
% For this optional (ungraded) exercise, you should implement the above strategy 
% for computing the learning curves in |learningCurve.m| and use the code below 
% to call your modified function and generate the plot.

lambda = 0.01;
[error_train, error_val] = learningCurve(X_poly, y, X_poly_val, yval, lambda);
plot(1:m, error_train, 1:m, error_val);

title(sprintf('Polynomial Regression Learning Curve (lambda = %f)', lambda));
xlabel('Number of training examples')
ylabel('Error')
axis([0 13 0 100])
legend('Train', 'Cross Validation')
%% 
% For reference, Figure 10 shows the learning curve we obtained for polynomial 
% regression with $\lambda= 0.01$. 
% 
% 
% 
% Your figure may differ slightly due to the random selection of examples. 
% 
% _You do not need to submit any solutions for this optional (ungraded) exercise._
%% Submission and Grading
% After completing various parts of the assignment, be sure to use the submit 
% function system to submit your solutions to our servers. The following is a 
% breakdown of how each part of this exercise is scored.
% 
% 
% 
% You are allowed to submit your solutions multiple times, and we will take 
% only the highest score into consideration.