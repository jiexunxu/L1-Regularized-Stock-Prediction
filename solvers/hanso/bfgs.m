function [x, f, g, H, iter, info] = bfgs(pars, options)
%BFGS The BFGS quasi-Newton minimization algorithm.
%   Simple call:   [x, f, g] = bfgs(pars) 
%   General call:  [x, f, g, H, iter, info] = bfgs(pars, options)
% 
%   Input parameters
%    pars is a required struct, with two required fields
%      pars.nvar: the number of variables
%      pars.fgname: string giving the name of function (in single quotes) 
%         that returns the function and its gradient at a given input x, 
%         with call   [f,g] = fgtest(x,pars)  if pars.fgname is 'fgtest'.
%         Any data required to compute the function and gradient may be
%         encoded in other fields of pars.
%    options is an optional struct, with no required fields
%      options.x0: each column is a starting vector of variables
%          (default: empty)
%      options.nstart: number of starting vectors, generated randomly
%          if needed to augment those specified in options.x0
%          (default: 10 if options.x0 is not specified)
%      options.maxit: max number of iterations
%          (default 100) (applies to each starting vector)
%      options.normtol: termination tolerance on gradient norm
%          (default: 1e-6) (applies to each starting vector)
%      options.fvalquit: quit if f drops below this value 
%          (default: -inf) (applies to each starting vector)
%      options.xnormquit: quit if ||x|| reaches this size
%          (default: inf) (applies to each starting vector)
%      options.cpumax: quit if cpu time in secs exceeds this 
%          (default: inf) (applies to total running time)
%      options.H0: initial inverse Hessian approximation 
%          (default: identity)
%      options.strongwolfe: 0 for weak Wolfe line search (default)
%                           1 for strong Wolfe line search
%      options.wolfe1: first Wolfe line search parameter 
%          (ensuring sufficient decrease in function value, default: 1e-4)
%      options.wolfe2: second Wolfe line search parameter 
%          (ensuring algebraic increase (weak) or absolute decrease (strong)
%           in projected gradient, default: 0.9)
%          ("strong" Wolfe line search is not usually recommended for use with
%           BFGS; it is very complicated and bad if f is nonsmooth, but
%           it could have advantages in some cases)
%      options.prtlevel: one of 0 (no printing), 1 (minimal), 2 (verbose)
%          (default: 1)
%
%   Output parameters 
%    x: each column is an approximate minimizer, one for each starting vector
%    f: each entry is the function value for the corresponding column of x
%    g: each column is the gradient at the corresponding column of x
%    H: cell array of the final BFGS inverse Hessian approximation matrices
%     (except that H is a matrix if there is only one starting vector)
%    iter: each entry is number of BFGS iterations used, for each starting vector
%    info: each entry is reason for termination, for each starting vector:
%     0: tolerance on gradient norm reached  
%     1: max number of iterations reached
%     2: f reached target value
%     3: norm(x) exceeded limit
%     4: cpu time exceeded limit
%     5: f is inf or nan at initial point
%     6: direction not a descent direction
%     7: line search bracketed minimizer but Wolfe conditions not satisfied
%     8: line search did not bracket minimizer: f may be unbounded below 
%
%   BFGS is normally used for optimizing smooth, not necessarily convex, 
%   functions, for which the convergence rate is generically superlinear.
%   Surprisingly, it often works very well for functions that are nonsmooth at
%   their minimizers, although the convergence rate is linear at best, and
%   the final inverse Hessian approximation is typically very ill conditioned.
%   Version 1.01, May 2009 (minor changes to Version 1.00, July 2006)
%   Send comments/bug reports to Michael Overton, overton@cs.nyu.edu,
%   with a subject header containing the string "bfgs".

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  HANSO, Hybrid Algorithm for NonSmooth Optimization
%%  Copyright (C) 2008  James Burke, Adrian Lewis, and Michael Overton
%%
%%  This program is free software: you can redistribute it and/or modify
%%  it under the terms of the GNU General Public License as published by
%%  the Free Software Foundation, either version 3 of the License, or
%%  (at your option) any later version.
%%
%%  This program is distributed in the hope that it will be useful,
%%  but WITHOUT ANY WARRANTY; without even the implied warranty of
%%  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%%  GNU General Public License for more details.
%%
%%  You should have received a copy of the GNU General Public License
%%  along with this program.  If not, see <http://www.gnu.org/licenses/>.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% parameter defaults
if nargin == 0
   error('bfgs: "pars" is a required input parameter')
end
if nargin == 1
   options = [];
end
options = setdefaults(pars,options);  % set default options
options = setx0(pars, options); % augment options.x0 randomly
cpufinish = cputime + options.cpumax;
prtlevel = options.prtlevel;
options = setwolfedefaults(options, 0);  % fields for Wolfe line search
if ~isfield(options,'H0')
    options.H0 = []; % not the identity, see bfgs1run.m
end
x0 = options.x0;
nstart = size(x0,2);
for run = 1:nstart
    if prtlevel > 0 & nstart > 1
        fprintf('bfgs: starting point %d\n', run)
    end
    options.cpumax = cpufinish - cputime; % time left
    [x(:,run), f(run), g(:,run), HH, iter(run), info(run)] = bfgs1run(x0(:,run), pars, options);
    % make exactly symmetric (too expensive to do inside optimization loop}
    H{run} = (HH + HH')/2; % make exactly symmetric (too expensive to do inside optimization loop}
    if cputime > cpufinish
        break
    end
end
if nstart == 1
    H = H{1};
end