classdef learnODE < Problem

  % Properties
  properties

    batch_counter   % batch_counter
    data            % training data
    inputs          % data inputs
    inputs_con      % data inputs for constraints
    mass_coeffs     % mass balance coefficients
    network         % neural network
    n_batches       % number of batches
    n_constraints   % number of constraints
    n_features      % number of features
    n_hidden_layers % number of hidden layers
    n_hidden_nodes  % number of nodes in each hidden layer
    n_outputs       % number of outputs
    n_samples       % number of samples for objective
    n_Constraint_samples
    set_constraint
    n_variables     % number of variables (learnable network values)
    outputs         % data outputs
    times_con
    x_last          % last primal variable
    x_table         % table of variables

  end

  % Methods
  methods (Access = public)

    % Constructor
    function self = learnODE(ins,outs,coeffs,S_set)

      % Initialize random number generator
      rng('default');

      % Set data
      self.inputs      = dlarray(ins,'CB');
      self.outputs     = outs;
      self.mass_coeffs = coeffs;

      % Set number of features and outputs
      self.n_features = size(ins,1);
      self.n_outputs  = size(outs,1);
      self.n_samples  = size(ins,2);

      % Set constraint data
      self.n_Constraint_samples = length(S_set);
      self.set_constraint = S_set;
      % self.n_constraints = m;
      % con_indices        = 1:(self.n_samples/self.n_constraints):self.n_samples;
      self.times_con     = dlarray(ins(1,S_set),'CB');
      self.inputs_con    = dlarray(ins(2:end,S_set),'CB');
      

      % Set number of batches
      self.n_batches     =  1;
      self.batch_counter = -1;

      % Set hidden layer sizes
      self.n_hidden_layers = 2; %2
      for i = 1:self.n_hidden_layers
        self.n_hidden_nodes(i) = 512; % 512
      end

      % Set number of optimization variables
      self.n_variables = self.n_features * self.n_hidden_nodes(1) + self.n_hidden_nodes(1);
      for i = 2:self.n_hidden_layers
        self.n_variables = self.n_variables + self.n_hidden_nodes(i-1) * self.n_hidden_nodes(i) + self.n_hidden_nodes(i);
      end
      self.n_variables = self.n_variables + self.n_hidden_nodes(end) * self.n_outputs + self.n_outputs;

      % Set weight and bias terms
      W{1} = zeros(self.n_hidden_nodes(1),self.n_features);
      b{1} = zeros(self.n_hidden_nodes(1),1);
      for i = 2:self.n_hidden_layers
        W{i} = zeros(self.n_hidden_nodes(i),self.n_hidden_nodes(i-1));
        b{i} = zeros(self.n_hidden_nodes(i),1);
      end
      W{self.n_hidden_layers+1} = zeros(self.n_outputs,self.n_hidden_nodes(end));
      b{self.n_hidden_layers+1} = zeros(self.n_outputs,1);

      % Initialize neural network
      self.network = initializeNeuralNetwork(self, W, b);

      % Set table value
      for i = 1:self.n_hidden_layers+1
        table_value{2*(i-1)+1,1} = W{i};
        table_value{2*(i-1)+2,1} = b{i};
      end

      % Set variable table
      self.x_table = table(self.network.Learnables.Layer, self.network.Learnables.Parameter, table_value, 'VariableNames', {'Layer', 'Parameter', 'Value'});

      % Initialize last primal variable
      self.x_last = zeros(self.n_variables,1);

    end

    % Initialize neural network
    function network = initializeNeuralNetwork(self, W, b)

      % Set input layer
      layers = [featureInputLayer(self.n_features)];

      % Loop through hidden layers
      for i = 1:self.n_hidden_layers
        layers = [layers
                  fullyConnectedLayer(self.n_hidden_nodes(i), 'Name', 'layer', 'Weights', W{i}, 'Bias', b{i})
                  tanhLayer]; % tanhLayer leakyReluLayer eluLayer]; %
      end

      % Set output layer
      layers = [layers
                fullyConnectedLayer(self.n_outputs,  'Name', 'layer', 'Weights', W{self.n_hidden_layers+1}, 'Bias', b{self.n_hidden_layers+1})];

      % Create neural network
      network = dlnetwork(layers);

    end % initializeNeuralNetwork

    % Compute forward solution
    function Y_best = computeForward(self,ins,x_best,f_best)

      % Set test inputs
      ins = dlarray(ins,'CB');

      % Set best iterate
      self.updateNetworkVariables(x_best);

      % Evaluate forward passes
      Y = forward(self.network, ins);

      % Extract data
      Y_best = extractdata(Y);

      % Save values
      save(f_best,"x_best","Y_best");

    end % computeForward

    % Evaluate constraint function and Jacobian value, equalities
    function [cE,JE] = constraintFunctionAndJacobianEqualities(self,network,inputs,times)

      % Evaluate forward pass for initial time
      Y = forward(network, [times; inputs]);

      % Evaluate gradient with respect to outputs
      gradY = dlarray(zeros(self.n_outputs,length(times)),'CB');
      HessY = dlarray(zeros(self.n_outputs,length(times)),'CB');
      for i = 1:self.n_outputs
        gradY(i,:) = dlgradient(sum(Y(i,:)),times,EnableHigherDerivatives=true);
      end
      for i = 1:self.n_outputs
        HessY(i,:) = dlgradient(sum(gradY(i,:)),times,EnableHigherDerivatives=true);
      end
      
      Hess_grad_Y = [HessY; gradY; Y];
      % Evaluate mass balance coefficients times derivatives
      CgradY = fullyconnect(Hess_grad_Y, self.mass_coeffs, zeros(1,1));

      % Augment constraint function value
      cEdl = [];
      for i = 1:self.n_outputs
        cEdl = [cEdl; sum(CgradY(i,:))/length(times)];
      end
      
      % Evaluate Jacobian
      for i = 1:self.n_outputs
        JEdl{i} = dlgradient(cEdl(i),network.Learnables);
      end
      % Evaluate constraint value
      cE = double(extractdata(cEdl));

      % Flatten
      for i = 1:self.n_outputs
        JE(i,:) = double(self.flattenGradient(JEdl{i}))';
      end

    end

    % Evaluate objective function and gradient value
    function [f,g] = objectiveFunctionAndGradient(self,network,inputs,outputs)

      % Evaluate forward passes
      Y = forward(network, inputs);

      % Evaluate deep learning objective function
      f = mse(Y - outputs, zeros(size(outputs)));

      % Evaluate deep learning objective gradient
      g = dlgradient(f,network.Learnables);

      % Set function value
      f = double(extractdata(f));

      % Set gradient value
      g = double(self.flattenGradient(g));

    end % objectiveFunctionAndGradient


    % Update network constraint sample set
    function updateNetworkConstraintSet(self,S)

    % Check if equal to last
      if isequal(S, self.set_constraint)
        return
      end

      self.n_Constraint_samples = length(S);
      self.set_constraint = S;
      % self.n_constraints = m;
      % con_indices        = 1:(self.n_samples/self.n_constraints):self.n_samples;
      self.times_con     = dlarray(self.inputs(1,S),'CB');
      self.inputs_con    = dlarray(self.inputs(2:end,S),'CB');

    end


    % Update network variables and constraints
    function updateNetworkVariables(self,x)

      % Check if equal to last
      if isequal(x,self.x_last) 
        return
      end

      % Set indices
      W_idx{1} = 1:(self.n_features * self.n_hidden_nodes(1));
      b_idx{1} = W_idx{1}(end) + [1:self.n_hidden_nodes(1)];
      for i = 2:self.n_hidden_layers
        W_idx{i} = b_idx{i-1}(end) + [1:(self.n_hidden_nodes(i-1)*self.n_hidden_nodes(i))];
        b_idx{i} = W_idx{i}(end) + [1:self.n_hidden_nodes(i)];
      end
      W_idx{self.n_hidden_layers+1} = b_idx{self.n_hidden_layers}(end) + [1:(self.n_hidden_nodes(end)*self.n_outputs)];
      b_idx{self.n_hidden_layers+1} = W_idx{self.n_hidden_layers+1}(end) + [1:self.n_outputs];

      % Separate variables
      W{1} = reshape(x(W_idx{1}), [self.n_hidden_nodes(1), self.n_features]);
      b{1} = reshape(x(b_idx{1}), [self.n_hidden_nodes(1), 1]);
      for i = 2:self.n_hidden_layers
        W{i} = reshape(x(W_idx{i}), [self.n_hidden_nodes(i), self.n_hidden_nodes(i-1)]);
        b{i} = reshape(x(b_idx{i}), [self.n_hidden_nodes(i), 1]);
      end
      W{self.n_hidden_layers+1} = reshape(x(W_idx{self.n_hidden_layers+1}), [self.n_outputs, self.n_hidden_nodes(end)]);
      b{self.n_hidden_layers+1} = reshape(x(b_idx{self.n_hidden_layers+1}), [self.n_outputs, 1]);

      % Set variable table
      for i = 1:self.n_hidden_layers+1
        self.x_table.Value{2*(i-1)+1,1} = dlarray(W{i});
        self.x_table.Value{2*(i-1)+2,1} = dlarray(b{i});
      end

      % Update network weights
      updateFunction = @(network,values) values;
      self.network = dlupdate(updateFunction, self.network, self.x_table);

    end

    % Flatten gradient
    function g = flattenGradient(self, g)

      % Reshape values
      for i = 1:self.n_hidden_layers+1
        g_W{i} = reshape(extractdata(cell2mat(g.Value(2*(i-1)+1))), [], 1);
        g_b{i} = reshape(extractdata(cell2mat(g.Value(2*(i-1)+2))), [], 1);
      end

      % Vectorize
      g = [];
      for i = 1:self.n_hidden_layers+1
        g = [g; g_W{i}; g_b{i}];
      end

    end

    % Constraint function, equalities
    function [cE,err] = evaluateConstraintFunctionEqualities(self,x)

      % Update network variables
      self.updateNetworkVariables(x);

      % Evaluate constraint function, equalities
      [cE,~] = dlfeval(@self.constraintFunctionAndJacobianEqualities,self.network,self.inputs_con,self.times_con);

      % No error
      err = false;

    end % evaluateConstraintFunctionEqualities

    % Constraint function, inequalities
    function [cI,err] = evaluateConstraintFunctionInequalities(self,~)

      % Evaluate constraint function, inequalities
      cI = [];

      % No error
      err = false;

    end % evaluateConstraintFunctionInequalities

    % Constraint Jacobian, equalities
    function [JE,err] = evaluateConstraintJacobianEqualities(self,x)

      % Update network variables
      self.updateNetworkVariables(x);

      % Evaluate constraint Jacobian, equalities
      [~,JE] = dlfeval(@self.constraintFunctionAndJacobianEqualities,self.network,self.inputs_con,self.times_con);

      % No error
      err = false;

    end % evaluateConstraintJacobianEqualities

    % Constraint Jacobian, inequalities
    function [JI,err] = evaluateConstraintJacobianInequalities(self,~)

      % Evaluate constraint Jacobian, inequalities
      JI = [];

      % No error
      err = false;

    end % evaluateConstraintJacobianInequalities

    % Hessian of the Lagrangian
    function [H,err] = evaluateHessianOfLagrangian(self,~,~,~)

      % Evaluate Hessian of Lagrangian
      H = eye(self.n_variables);

      % No error
      err = false;

    end % evaluateHessianOfLagrangian

    % Objective function
    function [f,err] = evaluateObjectiveFunction(self,x)

      % Update network variables
      self.updateNetworkVariables(x);

      % Evaluate objective function
      [f,~] = dlfeval(@self.objectiveFunctionAndGradient,self.network,self.inputs,self.outputs);

      % No error
      err = false;

    end % evaluateObjectiveFunction

    % Objective gradient
    function [g,err] = evaluateObjectiveGradient(self,x,type,~)

      % Update network variables
      self.updateNetworkVariables(x);

      % Check type
      if strcmp(type,"true")
        ins  = self.inputs;
        outs = self.outputs;
      else

        % Increment counter
        self.batch_counter = mod(self.batch_counter + 1, self.n_batches);

        % Set batch size
        batch_size = self.n_samples / self.n_batches;

        % Set times for gradient evaluation
        ins  = dlarray(self.inputs(:,self.batch_counter*batch_size+1:(self.batch_counter+1)*batch_size),'CB');
        outs = dlarray(self.outputs(:,self.batch_counter*batch_size+1:(self.batch_counter+1)*batch_size),'CB');

      end

      % Evaluate objective gradient
      [~,g] = dlfeval(@self.objectiveFunctionAndGradient,self.network,ins,outs);

      % No error
      err = false;

    end % evaluateObjectiveGradient

    % Initial point
    function x = initialPoint(self)

      % Set initial point
      %x = randn(self.n_variables,1);
      load("Y_best_1.mat","x_best");
      x = x_best;

    end % initialPoint

    % Name
    function s = name(self)

      % Set name
      s = 'learnODE';

    end % name

    % Number of constraints, equalities
    function mE = numberOfConstraintsEqualities(self)

      % Set number of constraints, equalities
      mE = self.n_constraints;

    end % numberOfConstraintsEqualities

    % Number of constraints, inequalities
    function mI = numberOfConstraintsInequalities(self)

      % Set number of constraints, inequalities
      mI = 0;

    end % numberOfConstraintsInequalities

    % Number of variables
    function n = numberOfVariables(self)

      % Set number of variables
      n = self.n_variables;

    end % numberOfVariables

  end

end