function [x, f, g, H, iter, info] = bfgs1run(x0, pars, options)
% make a single run of BFGS from one starting point
% intended to be called by bfgs only
% reference: Nocedal and Wright
% inputs:
%    x0: starting point
%    pars: defines function, see bfgs.m
%    options: see bfgs.m
% outputs: 
%    x: final iterate
%    f: final function value
%    g: final gradient
%    H: final inverse Hessian approximation
%    iter: number of iterations
%    info: reason for termination:
%     0: tolerance on gradient norm reached  
%     1: max number of iterations reached
%     2: f reached target value
%     3: norm(x) exceeded limit
%     4: cpu time exceeded limit
%     5: f is inf or nan at initial point
%     6: direction not a descent direction
%     7: line search bracketed minimizer but Wolfe conditions not satisfied
%     8: line search did not bracket minimizer: f may be unbounded below 

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
fgname = pars.fgname;
normtol = options.normtol;
fvalquit = options.fvalquit;
xnormquit = options.xnormquit;
cpufinish = cputime + options.cpumax;
maxit = options.maxit;
prtlevel = options.prtlevel;
strongwolfe = options.strongwolfe;
wolfe1 = options.wolfe1;
wolfe2 = options.wolfe2;
x = x0; 
[f,g] = feval(fgname, x, pars);
if size(g,2) > size(g,1) % error return is appropriate here
    error('gradient must be returned as a column vector, not a row vector')
end
gnorm = norm(g);
H0 = options.H0;
iter = 0;
if isempty(H0)
    H = eye(length(x0));
    scaleinit = 1; % so H is scaled before the first update
else
    H = H0;
    scaleinit = 0; % no scaling if H0 is provided
end
if f == inf % better not to generate an error return
    if prtlevel > 0
        fprintf('bfgs: f is infinite at initial iterate\n')
    end
    info = 5;
    return
elseif isnan(f)
    if prtlevel > 0
        fprintf('bfgs: f is nan at initial iterate\n')
    end
    info = 5;
    return
elseif gnorm < normtol
    if prtlevel > 0
        fprintf('bfgs: tolerance on gradient satisfied at initial iterate\n')
    end
    info = 0; 
    return
elseif f < fvalquit
    if prtlevel > 0
        fprintf('bfgs: below target objective at initial iterate\n')
    end
    info = 2;
    return
elseif norm(x) > xnormquit
    if prtlevel > 0
        fprintf('bfgs: norm(x) exceeds limit at initial iterate\n')
    end
    info = 3;
    return
end 
for iter = 1:maxit
    p = -H*g; % H approximates the inverse Hessian
    gtp = g'*p;
    if gtp >= 0 | isnan(gtp) % in rare cases, H could contain nans
       if prtlevel > 0
          fprintf('bfgs: not descent direction, quit at iteration %d, f = %g, gnorm = %5.1e\n',...
              iter, f, gnorm)
       end
       info = 6;
       return
    end
    gprev = g;
    if strongwolfe % strong Wolfe line search is optional
        [alpha, x, f, g, fail] = ...
            linesch_sw(x, f, g, p, pars, wolfe1, wolfe2, fvalquit, prtlevel);
    else % weak Wolfe line search is the default
        [alpha, x, f, g, fail] = ...
            linesch_ww(x, f, g, p, pars, wolfe1, wolfe2, fvalquit, prtlevel);
    end
    gnorm = norm(g);
    if prtlevel > 1
        fprintf('bfgs: iter %d: step = %5.1e, f = %g, gnorm = %5.1e\n', iter, alpha, f, gnorm)
    end
    if f < fvalquit % line search terminated when f drops below fvalquit
        if prtlevel > 0
            fprintf('bfgs: reached target objective, quit at iteration %d \n', iter)
        end
        info = 2;
        return
    elseif norm(x) > xnormquit % norm(x) is NOT checked inside line search
        if prtlevel > 0
            fprintf('bfgs: norm(x) exceeds limit, quit at iteration %d \n', iter)
        end
        info = 3;
        return
    end
    if fail == 1 % Wolfe conditions not both satisfied, quit
        if prtlevel > 0
           fprintf('bfgs: quit at iteration %d, f = %g, gnorm = %5.1e\n',...
               iter, f, gnorm)
        end
        info = 7;
        return
    elseif fail == -1 % function apparently unbounded below
        if prtlevel > 0
           fprintf('bfgs: f may be unbounded below, quit at iteration %d, f = %g\n', iter, f)
        end
        info = 8;
        return
    end
    if gnorm <= normtol 
        if prtlevel > 0
            fprintf('bfgs: gradient norm below tolerance, quit at iteration %d, f = %g\n', iter, f')
        end
        info = 0;
        return
    end
    if cputime > cpufinish
        if prtlevel > 0
            fprintf('bfgs: cpu time limit exceeded, quit at iteration %d\n', iter)
        end
        info = 4;
        return
    end
    s = alpha*p;
    y = g - gprev;
    sty = s'*y;    % successful line search ensures this is positive
    if sty > 0     % perform rank two BFGS update to the inverse Hessian H
        if iter == 1 & scaleinit % Nocedal and Wright recommend scaling I before first update
            H = (sty/(y'*y))*H; % equivalently, replace H on the right by I
        end
        rho = 1/sty;
        rhoHyst = rho*(H*y)*s';                                       % M = I - rho*s*y';
        H = H - rhoHyst' - rhoHyst + rho*s*(y'*rhoHyst) + rho*s*s';   % H = M*H*M' + rho*s*s';
    else % should not happen unless line search fails
        if prtlevel > 1
            fprintf('bfgs: *** sty <= 0, skipping BFGS update at iteration %d \n', iter) 
        end
    end
end % for loop
if prtlevel > 0
    fprintf('bfgs: %d iterations reached, f = %g, gnorm = %5.1e\n', maxit, f, gnorm)
end
info = 1;