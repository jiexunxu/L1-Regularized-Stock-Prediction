% This is the class for fast incremental time series prediction.
% 
% This object uses the following solvers to solve |Ax-b|_2^2+mu*|x|_1:
% 
% l1-regularized regression(http://www.stanford.edu/~boyd/l1_ls/):
% This solver is from Prof Boyd. We slighted modified the original code for our purpose. If the 
% condition number of the data is low, this is the fastest solver.
% 
% hanso(http://www.cs.nyu.edu/overton/software/hanso/):
% This is a very general purpose solver. We don't recommend using it 
% because it's slow.
%
% path-finding(http://sparselab.stanford.edu/):
% This solver is part of the SparseLab from Stanford. It's a very fast
% solver. Recommended when the user wants to get a large number of
% predicted results based on all possible mu values. When this solver is
% used, the value of mu is irrelevent.
% 
% Suppose the user has a time series matrix A of size t*n, and he or 
% she wishes to predict the values of series i at time t+1 and beyond.
% A is row ordered, which means each column of A is a series, and each row 
% of A is a time step. This is opposite to the what is explained in
% http://cs.nyu.edu/cs/faculty/shasha/papers/fps.d/explanation.htm.
%
% The user needs to construct an instance of this class by calling the
% constructor, obj=IncrementalPredict(...) with the appropriate parameters.
% See the documentation for the constructor of this class,
% IncrementalPredict(...), for more details.
%  
% Then the user call p=obj.predict(), which will store the desired
% predicted value in p. If solver is 'path-finding', then p is a column
% vector with 20 predicted values. See the documentation for
% solve_sparse_X() below for more details.
% 
% Finally when the actual values of each time series at time step t+1 are
% known (the new values should be in a row vector: call this vector v),
% the user call obj.update_fields(v) to update obj. 
% After calling update_fields(v), the user will be able to predict the value
% of series i at time step t+2.
%
% See sample_run.m included in this package for more details regarding the
% usage of this matlab object.
%
% author: Jiexun Xu
% reviewer: Carl Bosley
%
classdef IncrementalPredict < handle
    % properties of the class:
    %
    % A: The time series matrix, sliced by window_size if window_size is
    % non-zero. Last row in the time series matrix is not stored in A, but
    % in lastRowOfA. Each column of A should be a series, and each row of A
    % should be a time step
    %
    % i: The ith series(column) of the time series matrix that will be
    % predicted
    %
    % lastRowOfA: The last row of the time series matrix.
    %
    % mu: The mu value used in the l1-regularized optimization problem
    % |Ax-b|_2^2+mu*|x|_1. We recommend that the user call train_model()
    % often for stock prediction to get most up-to-date mu values, and then
    % call obj.update_mu(new_mu) to update the mu in this object.
    %
    % window_size: The number of most recent time steps that will be used
    % for computing the sparse X.
    %
    % cholA: The cholesky decomposition of [A b]'*[A b]. cholA will be
    % updated for each incoming new values of the time series.
    %
    % solver: A string that specify which solver to use. Currently accepted
    % strings are 'l1_ls', 'hanso', 'path_finding'.
    % 'l1_ls' is recommended for good data. 'path_finding' is
    % recommended for finding the predicted value of a large number of mu.
    %
    % remove_target_series: If true, then the ith column of A will be
    % removed; otherwise, keep that column in A. For stock prediction, we
    % recommend setting this parameter to true. If the user has a better
    % knowledge of the data and the user is sure that including the target
    % series will harm prediction, then the user can set it to false.
    %
    % b: The target series to be predicted. b is redundant if remove_target_series
    % is false, but when remove_target_series is true, b is essential to
    % store the historical values of the predicted series.
    %    
    % X0: An initial vector that serves as the input to some solvers.
    % It is initially set to a zero vector. The user can always change it by
    % calling update_X0(). We set X0 to be a zero vector by default. If the
    % user knows specific X0s that may help speeding up prediction, then we
    % recommend that the user set this X0 at appropriate time steps by
    % calling obj.update_X0(new_X0).
    %
    properties
        A
        i
        lastRowOfA
        mu
        window_size
        cholA
        solver
        remove_target_series
        b
        X0
    end
    
    methods        
        % Constructor.
        %
        % Parameters:
        %
        % A is the time series matrix. 
        %
        % i is the ith series that the user wants to predict. It's assumed that all values in A
        % are known. If there are t rows in A, then the object starts to
        % predict the value of series i at t+1.
        %
        % mu is the value of mu for |Ax-b|_2^2+mu*|x|_1
        %
        % window_size is the number of most recent time steps to use to
        % compute the sparse X.
        %
        % solver is the string that describe the particular solver to use.
        %
        % Returns: 
        %
        % An instance of this class
        %
        function obj=IncrementalPredict(A, i, mu, window_size, solver, remove_target_series)            
            m=size(A, 1);
            if window_size>=m
                error('input window size is not smaller than the number of rows in A');
            end
            
            if window_size~=0
                obj.A=A(m-window_size:m-1, :);
            else
                obj.A=A(1:m-1, :);
            end
            
            if window_size~=0
                b=A(m-window_size+1:m, i);
            else
                b=A(2:m, i);
            end
         
            obj.lastRowOfA=A(m, :);
            if remove_target_series
                obj.A(:, i)=[];
                obj.lastRowOfA(i)=[];
                obj.b=b;
            end
                        
            obj.i=i;            
            obj.mu=mu;
            cholOfA=[obj.A b]'*[obj.A b];
            obj.cholA=chol(cholOfA+eye(size(cholOfA, 1))*1e-14);
            obj.window_size=window_size;
            obj.solver=solver;
            obj.remove_target_series=remove_target_series;
            obj.X0=zeros(size(obj.cholA, 2)-1, 1);
        end
        
        % Compute the sparse linear combination vector X using the values
        % in A, and then compute the predicted value(s) of series i in the
        % next time step using lastRowOfA. 
        %
        % Parameters:
        %
        % obj: The object handle that refers to an IncrementalPredict
        % object
        %
        % Returns:
        %
        % If obj.solver is not 'path_finding', this function returns the
        % predicted value and the corresponding sparse X that computes the predicted value.
        %
        % If obj.solver is 'path_finding', this function returns an array
        % of 20 predicted values and the 20 corresponding Xs. See the
        % documentation for solve_sparse_X() for more details about these
        % 20 Xs. Suppose the user would like to do prediction, i.e, find an X to minimize 
        % |Ax-b|_2^2+0.052*|x|_1 and then use that X to compute the predicted value.
        %
        % After the user calls p=obj.predict() and get 20 values in p and X, the
        % user should choose X(5) to compute the predicted value, since X(5) is the solution that minimizes
        % |Ax-b|_2^2+0.05*|x|_1, with mu=0.05. This mu is closest to the
        % ideal mu=0.052 the user has in mind. Therefore, the user should
        % choose p(5) as the predicted value. If the user wants to solve
        % |Ax-b|_2^2+0.052*|x|_1 exactly, use 'l1_ls'. 
        %
        function [predicted_value, X]=predict(obj)   
            X=obj.solve_sparse_X();
            if strcmp(obj.solver, 'path_finding')
                predicted_value=zeros(20, 1);
                for i=1:20
                    predicted_value(i)=obj.lastRowOfA*X(:, i);
                end
            else                         
                predicted_value=obj.lastRowOfA*X;
            end
        end
        
        % Update the IncrementalPredict object obj when the actual values for each
        % time series are known.
        % 
        % Parameters:
        %
        % obj: The object handle that refers to an IncrementalPredict
        % object
        %
        % newA: The new values for each time series. newA should be a row
        % vector with one new value for each series. The dimension of newA
        % should equal to dimension of A.
        %        
        function update_fields(obj, newA)            
            new_row=[obj.lastRowOfA newA(obj.i)]';
            obj.A=[obj.A; obj.lastRowOfA];
            
            obj.lastRowOfA=newA; 
            if obj.remove_target_series
                obj.lastRowOfA(obj.i)=[];
                obj.b=[obj.b; newA(obj.i)];
            end
            window_size=obj.window_size;
            new_cholA=cholupdate(obj.cholA, new_row, '+');
            obj.cholA=new_cholA;                                          
            if window_size~=0
                if obj.remove_target_series
                    subtractedRow=[obj.A(1, :) obj.b(1)]'; 
                    obj.b(1)=[];
                else
                    subtractedRow=[obj.A(1, :) obj.A(2, obj.i)]'; 
                end
                obj.A(1, :)=[];         
                [new_cholA]=cholupdate(obj.cholA, subtractedRow, '-');
                obj.cholA=new_cholA;
            end           
        end
        
        % Update the X0 field in the IncrementalPredict object obj.
        % 
        % Parameters:
        %
        % obj: The object handle that refers to an IncrementalPredict
        % object
        % 
        % X0: The new vector that serve as the input for some solvers.
        %
        function update_X0(obj, X0)
            obj.X0=X0;
        end
        
        % Update the mu field in the IncrementalPredict object obj.
        % 
        % Parameters:
        %
        % obj: The object handle that refers to an IncrementalPredict
        % object
        % 
        % mu: The new mu in the model |Ax-b|_2^2+mu*|x|_1
        %
        function update_mu(obj, mu)
            obj.mu=mu;
        end
        
        % Solve for the sparse X in obj. Which solver to use depends on obj.solver
        %
        % Parameters:
        %
        % obj: The object handle that refers to an IncrementalPredict
        % object
        %
        % Returns:
        %
        % If the solver is not 'path_finding', then this function returns
        % the X by using the corresponding solver in obj.solver and related
        % parameters.
        %
        % If the solver is 'path_finding', then the solver SolveLasso_mod.m
        % will return a lot of Xs. Each of these X corresponds to a mu
        % value. That is, X_i is the solution that minimizes |Ax-b|_2^2+mu_i*|x|_1 
        % for mu_i. The total number of Xs depends on the dimension of A.
        % This function, however, returns 20 Xs regardless of the dimension
        % of A. These 20 Xs are determined in the following way: 
        %
        % The function puts these (X_i, mu_i) pairs into 20 bins.X_i is put 
        % into Bin i(1<=i<=20) if its corresponding mu_i satisfy 
        % 0.01*(i-1)<=mu_i<0.01*i. X_i is discarded if its corresponding mu_i is larger
        % than 0.2. For each bin, the X_i inside the bin with the maximum
        % mu_i among all other mu_i's in the same bin is returned. We
        % return the X_i with maximum mu_i in bin i such that X_i is
        % closest to the solution that minimizes |Ax-b|_2^2+(0.01*i)*|x|_1.
        % If a particular bin doesn't have any X, then return the zero vector for this bin.
        % 
        % Therefore, the function returns an array of Xs such that the ith X, X_i, is
        % the solution that minimizes |Ax-b|_2^2+mu_i*|x|_1, where mu_i is 
        % very close to or equal to 0.01*i. 
        %
        function X=solve_sparse_X(obj)
            m=size(obj.cholA, 2)-1;
            A=obj.cholA(:, 1:m);
            b=obj.cholA(:, m+1);
            solver=obj.solver;
            mu=obj.mu;  
            % solves |Ax-b|_2^2+mu*|x|_2^1 by using l1_ls_mod
            if strcmp(solver, 'l1_ls')                
                X=l1_ls_mod(A, obj.X0, b, mu);
            % solves |Ax-b|_2^2+mu*|x|_2^1 by using hanso
            elseif strcmp(solver, 'hanso')
                pars=struct('nvar', size(A, 2), 'fgname', 'basic_fpc', 'A', A, 'AA', A'*A, 'Ab', A'*b, 'b', b, 'mu', mu);   
                options=struct('prtlevel', 0);                
                options.x0=obj.X0;
                X=hanso(pars, options);            
            % solves |Ax-b|_2^2+mu*|x|_2^1 for all possible mu by using SolveLasso_mod. 
            elseif strcmp(solver, 'path_finding')
                [sols, mus]=SolveLasso_mod(A, b, size(A, 2), 'lasso', 10*size(A, 2), 0, 0, 1, false);
                % Put each X into its corresponding bin, and return the X
                % for each bin such that X's corresponding mu is largest in
                % that bin.
                X=zeros(size(sols, 1), 20);
                for i=size(sols, 2):-1:1
                    bin_index=ceil(mus(i)*100);
                    if bin_index<=size(X, 2)
                        X(:, bin_index)=sols(:, i);
                    end
                end                
            else
                error('undefined solver'); 
            end
        end       
    end
end   
