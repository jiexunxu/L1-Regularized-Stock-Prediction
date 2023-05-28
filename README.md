# Introduction

Given a collection of (ideally sparsely correlated) time series, FPS provides functions to make fast and accurate predictions of the future value of a particular time series by using the l1-regularized least squares model and cholesky decomposition and cholesky update. Interested users should carefully read the documentations and integrate our functions to their Matlab projects.

The sample data contained in this project is the difference of the stock prices for the 30 Dow & Jones stocks in Feburary, 2008. Each column is a particuluar Dow & Jones stock. Each row is the difference of the stock price between the price of each stock at current time and the price of the corresponding stock one minute earlier. 

# Installation

FPS works with Windows/Unix/Linux/Mac.

First, You need to install Matlab. 

Next, you need to set Matlab's PATH variable. Open Matlab, click File -> Set Path... Then click "Add with Subfolders..." on the upper left corner. A prompt will pop up asking you which folder and its subfolders will be added to Matlab's search path. Go to the directory where you have stored FPS, select the FPS folder, and click "OK". The matlab search path has been set up. If you wish to save this new PATH variable, click "Save" on the bottom left corner. You should be able to use our software next time you start Matlab. 

# Run

FPS is not an executable project. Instead FPS is a package that contains a function to train our l1-regularized least squares model, and a Matlab object that performs predictions on a particular time series given other time series and itself.

There are two Matlab programs that should be of interest to you. train_model.m is the function that suggests the user possibly good parameters to use in actual predictions., and IncrementalPredict.m is a Matlab object that makes future predictions on a particular time series. 

We assume that the user already have some initial data for each series so that we can first run train_model.m on these data to obtain some parameters before we any actual prediction. We also assume that the user will constantly obtain new values for each series from some data source. Both the initial data and the incoming new data are likely to be part of a large system, so we expect that both of train_model.m and IncrementalPredict.m will be embeded in that system. In fact there's no "main" method in this package. We only provide a Matlab class IncrementalPredict.m for doing prediction and a Matlab function train_model.m that output suggested parameters based on the training_data. We assume that interested user will embed this class and this function into a larger Matlab project.

To get a feeling of how to use IncrementalPredict.m and train_model.m, see the file sample_run.m. To run this program, type "sample_run;"(excluding the double quotes(")) at Matlab's command prompt. The default parameters to run this program should be appropriate for a start.

This sample program will make predictions for a particular series in a given time interval. The program assumes that the input data is volatile; therefore, it revokes the train_model function every 100 time steps to get the most up-to-date parameters for actual predictions.

We suggest that interested users carefully read the documentations in sample_run.m, IncrementalPredict.m and train_model.m to understand how IncrementalPredict.m and train_model.m may be applied to their Matlab project.

WARNING: This demo program is going to take a long time (a few hours to about a day) to finish if the difference of start_time and end_time is large (say larger than 3000).

# License

FPS is a free software. You may use it "as is" for educational or commercial purposes. You must understand however that it is experimental software so may have bugs.
