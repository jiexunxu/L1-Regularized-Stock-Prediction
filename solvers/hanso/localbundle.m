function [x, f, g, dnorm, xbundle, gbundle, w] = localbundle(x, f, g, pars, options)
% local bundle method starting at x with steepest descent and
% adding gradients returned by weak Wolfe line search even if line search
% fails, terminating when d is generated with ||d<< <= options.normtol
% (success), or, after a line search, if beta*||d|| > options.evaldist, 
% where beta is the step taken along d to where the new gradient is
% evaluated, or options.maxit iterations exceeded.
% called by hanso

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
normtol = options.normtol; % tolerance on norm(d)
evaldist = options.evaldist; % max distance between successive gradient evals
prtlevel = options.prtlevel;
maxit = options.maxit;
nvar = pars.nvar; 
cpufinish = cputime + options.cpumax;
% tell quadprog to keep quiet
options.qp_options = optimset('Display','off','Diagnostics','off'); 
bundlesize = 1;
xbundle = x;
gbundle = g;
w = 1;
d = -g;
dnorm = norm(d);
if dnorm < normtol
    fprintf('localbundle: norm(g) below tolerance at initial iterate\n')
    return
end
success = 0;
for iter = 1:maxit
    gtd = g'*d;  
    if gtd >= 0 | isnan(gtd)
        if prtlevel > 0
            fprintf('localbundle: quit at iter %d since not descent direction, f = %g, dnorm = %5.1e\n',...
                iter, f, dnorm)
        end
        return
    end
    % setting both Wolfe parameters to 0, so simple descent must be
    % obtained and sign of directional derivative must change if
    % alpha is returned nonzero
    % and setting fvalquit to -inf, as not relevant to local bundle
    [alpha, x, f, g, fail, beta, gbeta] = ... 
        linesch_ww(x, f, g, d, pars, 0, 0, -inf, prtlevel);
    if prtlevel > 1
        fprintf('localbundle: #grads = %d, f = %g, alpha = %5.1e, beta = %5.1e, dnorm = %5.1e\n',...
            size(gbundle,2), f, alpha, beta, dnorm);
    end
    % add gradient encountered in line search (if Wolfe conditions were
    % not satisfied, this is the gradient at beta, the RIGHT end of an
    % interval bracketing a point where the Wolfe conditions are satisfied)
    if any(isnan(gbeta)) % includes case beta = inf
        if prtlevel > 1
            fprintf('localbundle: quit at iter %d, f = %g since new gradient is nan\n', ...
                iter, f)
        end
        return
    else
        xbundle = [xbundle (x + beta*d)];
        gbundle = [gbundle gbeta];
    end
    % check termination condition AFTER adding gradient since may need
    % it for a BFGS update in HANSO
    % terminate if new gradient was evaluated too far away from x0, or
    % if beta = inf, so no new gradient was added to bundle
    if beta*dnorm > evaldist
        if prtlevel > 0
            fprintf('localbundle: quit at iter %d since step too big, f = %g, alpha = %5.1e, beta = %5.1e, dnorm = %5.1e\n',...
                iter, f, alpha, beta, dnorm)
        end
        return
    end
    % get next search direction by solving QP: d is smallest vector in
    % convex hull of negative gradients
    [w,d] = qpsubprob(gbundle, options); 
    dnorm = norm(d);
    if dnorm < normtol
        if prtlevel > 0
            fprintf('localbundle: verified optimality within tolerance at iter %d, f = %g, dnorm = %5.1e\n',...
                iter, f, dnorm)
        end
        return
    end
    if cputime > cpufinish
        if prtlevel > 0
            fprintf('quit since cpu time limit exceeded\n')
        end
        return
    end
end
if prtlevel > 0
    fprintf('localbundle: %d iterates reached, f = %g, dnorm = %5.1e\n', maxit, f, dnorm)
end