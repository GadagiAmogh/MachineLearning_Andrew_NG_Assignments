function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Feed Forward Propogation===================
y_out = zeros(num_labels, m); % y_out has the dimensions K(no of units in output layer) X M (no of examples). In our case it is 10 X 5000
for i =1:m;
    y_out(y(i), i)=1;
end
a1 = X; % a1 is the value of the input to the hidden layer to calculate a2
a1 = [ones(size(X, 1), 1) X]; %adding bias term and setting it to a0(lst yer = 1) = 1
a2 = sigmoid(Theta1 * a1'); %calculating the output of 2nd layer which will be the input to the 3rd layer
a2 = [ones(1, size(a2,2)); a2]; % adding bias term and setting it to a0(2nd layer = 1) = 1
a3 = sigmoid(Theta2 * a2); %calucalting the output of the whole neural network.

J = (-1/m) * (sum(sum(y_out .* log(a3) + (1-y_out) .* log(1-a3)))); % cost function is the the sqrd error of the output and input, hence a3 is is to be calculated from feed froward propogation.

Regularization = lambda/(2*m) * ((sum(sum(Theta1(:, 2:end).^2)) + sum(sum(Theta2(:, 2:end).^2)))); % regularizing to make sure every pixel (attribute) adds a bit to the output of the model

J = J + Regularization; 

for i = 1:m;
    a_1 = X(i,:); % Here we take the i th example for training.
    a_1 = [1 a_1]; %adding bias term and setting it to a0(lst yer = 1) = 1
    z_2 = Theta1 * a_1'; %defining z_2
    a_2 = sigmoid(z_2); %calculating the output of 2nd layer which will be the input to the 3rd layer
    a_2 = [1; a_2]; % adding bias term and setting it to a0(2nd layer = 1) = 1
    z_3 = Theta2 * a_2; %defining z_3
    a_3 = sigmoid(z_3); %calucalting the output of the whole neural network.
    
    delta_3 = a_3 - y_out(:,i); % Output Layer Error Term
    delta_2 = Theta2' * delta_3 .* sigmoidGradient([1; z_2]);
    delta_2 = delta_2(2:end); %Note that you should skip or remove  delta2(0) (bias term error)
    DELTA_2 = delta_3 * a_2'; % Back Propogating the Error term
    DELTA_1 = delta_2 * a_1; % Back Propogating the Error term
    Theta2_grad = Theta2_grad + DELTA_2; % Accumulating the gradient for all the examples for delta2
    Theta1_grad = Theta1_grad + DELTA_1; % Accumulating the gradient for all the examples for delta1
end

Theta2_grad = (1/m) * Theta2_grad; %Obtain the (unregularized) gradient for the neural network 
Theta1_grad = (1/m) * Theta1_grad; %cost function by dividing the accumulated gradients by m.

Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + (lambda/m) * Theta1(:,2:end); % Removing the bias term i.e. first column from Theta1_grad and Theta2_grad which defined in this function earlier.
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + (lambda/m) * Theta2(:,2:end); % Regularizing the gradient.

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
