% This program is a demonstration of how to use the Matlab files train_model.m
% and IncrementalPredict.m. All required parameters are in the
% configuration file sample_config.txt. Refer to
% configuration_file_instruction.txt for details regarding the meaning of
% each parameter. 
%
% To run this program, modify the parameters in the parameter file, and
% then type "sample_run;"(excluding the double quotes(")) at Matlab's command
% prompt.
%
function sample_run
    % These are the parameters ncessary to successfully run our simulation.
    % The default values should be appropriate for this program, but feel free to change them.
    %
    % target_series: The series to be predicted. Should not be larger than
    % the total number of series in full_data.
    % 
    % solver: The convex optimization solver used for prediction
    %
    % remove_target_series: Whether or not the target series will be
    % removed in prediction.
    %
    % start_step: the time step that prediction will started. Should not be
    % larger than the maximum number of time steps available in full_data.
    %
    % end_step: the time step that prediction will finish. Should not be
    % larger than the maximum number of time steps available in full_data.
    %
    fid=fopen('sample_config.txt');    
    % parse the parameters
    data_filename=fgetl(fid);
    solver=fgetl(fid);   
    numeric_values=fscanf(fid, '%d %d %d %d %d %d');
    target_series=numeric_values(1);
    start_step=numeric_values(2);
    end_step=numeric_values(3);
    window_size=numeric_values(4);
    remove_target_series=numeric_values(5); 
    training_frequency=numeric_values(6);
    fclose(fid);
    
    full_data=load(data_filename);
    % feed all historical data (from time step 1 to start_step) to the
    % IncrementalPredict object predictor
    initA=full_data(1:start_step-1, :);
    predictor=IncrementalPredict(initA, target_series, 1, window_size, solver, remove_target_series);
    
    % keep track of the latest "start_step" number of data in
    % training_data. training_data will be used for training every 100 time
    % steps
    training_data=initA;
    
    % pp is the sum of predicted_value squared, where predicted_value is
    % the predicted value of the particular series at each time step
    % vv is the sum of true_value squared, where true_value is the actual
    % value of the particular series at each time step.
    % pv is the sum of predicted_value times true_value
    % these three variables will be used to evaluate how much profit we've
    % made, and normalize this profit.
    pp=0;
    pv=0;
    vv=0;  
    % prediction_count counts the number of predictions made between each
    % training session.
    prediction_count=training_frequency;
    for time_step=start_step-1:end_step-1    
        % every training_frequency time steps, train the model and update the mu value in
        % predictor. We've ignored the suggested window_size because our
        % experiments show that many times we don't need to worry about
        % both window_size and mu. We fix
        % window_size to be 100 (this window_size is used both in training
        % and in prediction) by default, and only train mu.        
        if prediction_count>=training_frequency
            training_pars=struct('training_data', training_data, 'solver', 'path_finding', 'window_sizes', window_size);
            suggested_pars=train_model(training_pars);
            predictor.update_mu(suggested_pars(1));
            prediction_count=0;
        else
            prediction_count=prediction_count+1;
        end
        
        % predict the value of series "target_series" at time step
        % time_step+1
        predicted_value=predictor.predict(); 
        true_value=full_data(time_step+1, target_series);
        fprintf('The predicted value for series %d at time step %d is %d. The actual value is %d\n', target_series, time_step+1, predicted_value, true_value);
        
        pp=pp+predicted_value*predicted_value;
        vv=vv+true_value*true_value;
        pv=pv+predicted_value*true_value;
        
        % This is the new data for each series at the next time step. In
        % reality, we assume that the user has some Matlab function that
        % gives these new values. 
        new_data=full_data(time_step+1, :);
        % Update the properties in predictor by giving the new data
        predictor.update_fields(new_data);
        
        % Update training_data. Remove the oldest data in training_data,
        % and append the latest data to the end of training_data.
        training_data(1, :)=[];
        training_data=[training_data; new_data]; %#ok<AGROW>
    end    
    fprintf('The total profit (normalized) you have made is %2.2f percent of what you have invested', pv*100/sqrt(pp*vv));
end