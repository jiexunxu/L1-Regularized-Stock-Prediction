% Given a piece of training data and other input parameters, this function
% computes the total profit (normalized) earned, and suggest a parameter
% set that is most likely to be the most suitable parameter for training.
%
% For simplicty, we suggest that the user fix the window_size and train mu
% only. Our research has shown that even if we fix the window_size at a
% reasonable value and only train mu, we can still obtain reasonably good
% results on the test data. We suggest that the user fix window_size to be
% 100. We also suggest the user to choose the solver 'path-finding'. This is
% the simplest way for training. Our experiments haven't prove that such
% simple methods would significantly lose prediction accuray. 
% We also suggest that the target series be retained, unless the user has 
% a strong reason to remove the target series.
%
% Parameters:
%
% input_parameters is a struct that must have the following required
% fields:
% 'training_data': The data used for training. Such data must follow this
% format: The data is an m by n matrix. Each column represents a series,
% and each row represents a time step. So such a matrix contains n series,
% each of which has values on m time steps.
% 'solver': The specific solver fed to IncrementalPredict objects.
% Currently supported solvers are 'l1_ls', 'hanso', 'path_finding'.
% 'window_sizes': A column vector of window sizes to be used. Typically
% window_sizes ranges from 20 to 400. 
%
% The following are optional fields:
% 'mus': The mu values used for training. 'mus' is necessary if solver is
% 'l1_ls' or 'hanso', but not required for 'path_finding'. mus must be a
% column vector.
% 'remove_target_series': Whether or not the series to be predicted will
% be removed. By default it's set to false.
% 'top_percents': A column vector of percentages used for analysis in
% evaluate_prediction_result(). See the documentation for that function for
% details regarding top_percents.
%
% Returns:
%
% suggested_parameters: See the documentation for the function
% evaluate_prediction_result() for more details.
%
% prediction_result: See the documentation for the function
% evaluate_prediction_result() for more details.
%
% average_profit: See the documentation for the function
% model_prediction() for more details.
%
function [suggested_parameters, prediction_result, average_profit]=train_model(input_parameters)
    A=input_parameters.training_data;
    solver=input_parameters.solver;
    window_sizes=input_parameters.window_sizes;
    mus=0;
    if isfield(input_parameters, 'mus')
        mus=input_parameters.mus;
    end    
    remove_target_series=false;
    if isfield(input_parameters, 'remove_target_series')
        remove_target_series=input_parameters.remove_target_series;
    end
    top_percents=0.05;
    if isfield(input_parameters, 'top_percents')
        top_percents=input_parameters.top_percents;
    end
    
    % perform predictions on the training data. Store the prediction
    % results in prediction_result
    prediction_result=model_prediction(A, mus, window_sizes, solver, remove_target_series);
    % analyze the prediction results and suggest the user good parameter
    % on the test data. The average profit for each profit is also returned
    % as an optional parameter if the user is interested
    [suggested_parameters, average_profit]=evaluate_prediction_result(prediction_result, top_percents);
end

% This function predicts the values of all series in A from time step
% window_size+2 to the last time step in A, for each window_size in
% 'window_sizes', and for each mu in 'mus'. Every interval_length time
% steps, the total profit made during this interval_length for all series
% is summed up, normalized, and stored in the corresponding entry in
% prediction_result. prediction_result is finally returned for future
% analysis.
%
% Parameters:
%
% A is the training data. 
%
% mus is the array of mu values used in the model |Ax-b|_2^2+mu*|x|_1.
% mus must be a column vector. Typically mu ranges from 0 to 0.5. If solver
% is 'path_finding', then this mus parameter is ignored.
%
% window_sizes is the array of window sizes used in cutting out data that
% is too historical to be used for prediction. window_sizes must be a column
% vector. Typicall each window_size ranges from 50 to 400.
%
% solver is the solver that will be used for prediction. Currently
% supported solvers are 'l1_ls_mod', 'hanso', 'path_finding'.
%
% remove_target_series is the parameter that decides whether the target
% series (the series to be predicted) will be removed from the data.
%
% Returns:
%
% The result of the prediction stored in prediction_result. The format of
% prediction_result is that each row represents the total profit
% (normalized) earned under a particular mu and a particular window_size.
% Suppose we have made p_i predictions in total, where each prediction has
% an associated true_value v_i. Then the normalized profit is
% sum(p_i*v_i)/sqrt(sum(p_i*p_i)*sum(v_i*v_i)). The first column of
% prediction_result stores sum(p_i*p_i), the second column stores
% sum(v_i*v_i), the third column stores sum(p_i*v_i), the fourth column
% stores each mu, and the fifth column stores each window_size.
%
function prediction_result=model_prediction(A, mus, window_sizes, solver, remove_target_series)
    m=size(A, 1);
    n=size(A, 2); 
    % If solver is 'path_finding', ignore input mus. 
    if strcmp(solver, 'path_finding')
        mus=(0.01:0.01:0.2)';        
    end
    prediction_result=zeros(size(window_sizes, 1)*size(mus, 1), 5);            
    
    % Store the mu and window_size value at the end of each row of
    % prediction result.
    for p=1:size(window_sizes, 1)
        for q=1:size(mus, 1)
            index=(p-1)*size(mus, 1)+q;
            prediction_result(index, 4)=mus(q);
            prediction_result(index, 5)=window_sizes(p);
        end
    end
    
    % Set mus back to zero if solver is 'path_finding'. This avoids
    % redundant loops on same mu values below.
    if strcmp(solver, 'path_finding')
        mus=0;
    end
    
    % Loop through each window size value
    for window_size_index=1:size(window_sizes, 1)
        window_size=window_sizes(window_size_index);       
        A0=A(1:window_size+1, :);
        
        % Loop through each mu value
        for mu_index=1:size(mus, 1);
            mu=mus(mu_index);
            objs=[];
            % Create an array of n IncrementalPredict objects
            for i=1:n
                objs=[objs IncrementalPredict(A0, i, mu, window_size, solver, remove_target_series)];  %#ok<AGROW>
            end
            
            % pp is sum(p_i*p_i), vv is sum(v_i*v_i), pv is sum(p_i*v_i).
            % pp and vv will be of dimension 20 by 1 if solver is
            % 'path_finding'
            pp=0;
            vv=0;
            pv=0;
            if strcmp(solver, 'path_finding')
                pp=zeros(20, 1);
                pv=zeros(20, 1);
            end   
            
            % start predicting from time step window_size+2 to m
            for k=1+window_size:m-1
                for i=1:n
                    predicted_value=objs(i).predict();
                    new_values=A(k+1, :);
                    objs(i).update_fields(new_values);
                    true_value=A(k+1, i);
                    pp=pp+predicted_value.*predicted_value;                                     
                    vv=vv+true_value*true_value;
                    pv=pv+predicted_value*true_value;   
                end                
            end
            
            % store the results in prediction_result
            if ~strcmp(solver, 'path_finding')
                prediction_result((window_size_index-1)*size(mus, 1)+mu_index, 1)=pp;
                prediction_result((window_size_index-1)*size(mus, 1)+mu_index, 2)=vv;
                prediction_result((window_size_index-1)*size(mus, 1)+mu_index, 3)=pv;
            else
                for bin_index=1:20
                    prediction_result((window_size_index-1)*20+bin_index, 1)=pp(bin_index);
                    prediction_result((window_size_index-1)*20+bin_index, 2)=vv;
                    prediction_result((window_size_index-1)*20+bin_index, 3)=pv(bin_index);
                end
            end
        end                
    end
end

% This function analyzes the prediction result by computing the normalized
% profits for each parameter set, pick the top few percents of the profits
% designated by the parameter 'top_percents' and their associated parameter
% sets, normalize these parameters, compute the mean of these parameters,
% and return the parameter set that is closest to this mean.
%
% Parameters:
%
% prediction_result is the prediction result obtained from the function
% model_prediction().
%
% top_percents is an array of percentages. For each percentage p in
% top_percents, the best p percent of the profits will be picked out along
% with their parameters. 
%
% Returns:
%
% suggested_parameters is an array of suggested parameters for each
% percentage in top_percents. Row i of suggested_parameters corresponds to
% top_percents(i). The first column of suggested_parameters is the
% suggested mu, and the second column is the suggeested window_size.
%
% average_profit is an optional output that tells the user the normalized
% profit made by each parameter set and the associated parameters. The
% first column of average_profit is the normalized profit, the second
% column is the associated mu, and the third column is the associated
% window_size.
%
function [suggested_parameters, average_profit]=evaluate_prediction_result(prediction_result, top_percents) 
    m=size(prediction_result, 1);
    average_profit=zeros(size(m, 4));
    % compute the normalized profit for each parameter set and store them
    % in average_profit
    for i=1:m              
        average_profit(i, 1)=prediction_result(i, 3)/sqrt(prediction_result(i, 1)*prediction_result(i, 2));
        if prediction_result(i, 1)==0 || prediction_result(i, 2)==0
            average_profit(i, 1)=0;
        end
        average_profit(i, 2)=i;            
        average_profit(i, 3)=prediction_result(i, 4);
        average_profit(i, 4)=prediction_result(i, 5);
    end 
    % sort the columns of average_profit in descending order, using the
    % profits as the keys
    average_profit=sortrows(average_profit, -1);
    suggested_parameters=zeros(size(top_percents, 1), 2);
    % find a suggested parameter for each percentage in top_percents
    for k=1:max(size(top_percents, 1), size(top_percents, 2))
        percent=top_percents(k);
        cutting_cap=ceil(m*percent);
        pars=average_profit(1:cutting_cap, 3:4);
        [suggested_parameters(k, 1), suggested_parameters(k, 2)]=find_median_parameter(pars);
    end 
    % remove the second column of average profit since it's not interesting
    % to us. It's only an array of shuffled index number.
    average_profit(:, 2)=[];
end

% This function finds the parameter in pars that is closest to the mean of
% all normalized parameters in pars.
%
% Parameters:
%
% pars is a m by 2 matrix that stores mu in its first column, and
% window_size in its second column.
%
% Returns:
%
% optimal_mu is the suggested mu
%
% optimal_window_size is the suggested window_size
%
function [optimal_mu, optimal_window_size]=find_median_parameter(pars)
    % retain a copy of the input pars
    pars_copy=pars;    
    m=size(pars, 1);
    % computes the mean and standard deviation of mu and window_size
    mean1=sum(pars(:, 1))/m;
    mean2=sum(pars(:, 2))/m;  
    SD1=std(pars(:, 1));
    SD2=std(pars(:, 2));
    
    % normalize all parameters. The normalization goes as follows: Suppose
    % we have an array of parameters A. We compute their mean M and their
    % standard deviation SD. Then A(i)_normalized=(A(i)-m)/SD. If SD is
    % zero, then we just let A(i)_normalized=1. We normalize mus and
    % window_sizes seperately.
    if SD1~=0 
        for i=1:m
            pars(i, 1)=(pars(i, 1)-mean1)/SD1;           
        end
    else
        pars(:, 1)=ones(m, 1);
    end
    
    if SD2~=0 
        for i=1:m
            pars(i, 2)=(pars(i, 2)-mean2)/SD2;           
        end        
    else
        pars(:, 2)=ones(m, 1);
    end              
    % computes the new mean of normalized mus and window_sizes
    mean1=sum(pars(:, 1))/m;
    mean2=sum(pars(:, 2))/m;
    
    % find the parameter set in pars that is closest to the mean of both
    % parameters (i.e closest to the point (mean1, mean2))
    optimal_index=-1;
    min_dist=Inf;
    for i=1:size(pars, 1)
        dist=sqrt((mean1-pars(i, 1))^2+(mean2-pars(i, 2))^2);    
        if dist<min_dist
            min_dist=dist;
            optimal_index=pars(i, 2);
        end
    end
    % return the corresponding suggested parameter
    optimal_mu=pars_copy(optimal_index, 1);
    optimal_window_size=pars_copy(optimal_index, 2);
end