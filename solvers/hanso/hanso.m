function [x, f, loc, Y, G, w, H] = hanso(pars, options);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
%  Minimization algorithm intended for nonsmooth, nonconvex functions,
%   but also applicable to functions that are smooth, convex or both.
%  [X,F] = HANSO(PARS) returns approximate minimizer and function value 
%   of function with PARS.NVAR variables coded in mfile PARS.FGNAME.
%  [X,F,LOC] = HANSO(PARS,OPTIONS) allows specification of options and
%   also returns an approximate Local Optimality Certificate.
%  [X,F,LOC,Y,G,W,H] = HANSO(PARS,OPTIONS) returns additional data 
%   supporting the approximate Local Optimality Certificate.
% 
%   Input parameters
%    pars is a required struct, with two required fields
%      pars.nvar: the number of variables
%      pars.fgname: string giving the name of m-file (in single quotes) 
%         that returns the function and its gradient at a given input x, 
%         with call   [f,g] = fgtest(x,pars)  if pars.fgname is 'fgtest'.
%         Any data required to compute the function and gradient may be
%         encoded in other fields of pars. The user does not have to worry
%         about the nondifferentiable case or identify subgradients. 
%         The basic assumption is that the nondifferentiable case arises
%         with probability zero, and in the event that it does occur, it is
%         fine to return the gradient of the function at a nearby point.
%         For example, the user does not have to worry about ties in "max".
%    options is an optional struct, with no required fields
%      options.x0: columns are one or more starting vector of variables, 
%          used to intialize the BFGS phases of the algorithm
%          (default: generated randomly)
%      options.normtol: termination tolerance for smallest vector in 
%          convex hull of bundled or sampled gradients
%          (default: 1e-4)
%      options.evaldist: evaluation distance for gradients in each
%           local bundle phase and for the final gradient sampling phase
%          (default: 1e-4)
%      options.phasenum: vector of 3 quantities: number of each of
%          of the 3 phases (BFGS, local bundle, gradient sampling)
%          (default: [3 3 3])
%      options.phasemaxit: vector of 3 quantities: max iteration limit in 
%          each of the 3 phases (BFGS, local bundle, gradient sampling)
%          (default: [1000, pars.nvar + 10, 100])
%      options.ngrad: number of gradients used in gradient sampling phases
%          (default: min(100, pars.nvar + 10, 2*pars.nvar))
%      options.fvalquit: quit if f drops below this value 
%          (default: -inf)
%      options.cpumax: quit if cpu time in seconds exceeds this
%          (default: inf)
%      options.quadprog: 1 if quadprog is in path, -1 if not, 0 if not sure
%          (only the BFGS phases are run if quadprog is not in path)
%          (default: 0)
%      options.prtlevel: print level, 0 (no printing), 1, or 2 (verbose)
%          (default: 1)
%
%   Output parameters 
%    x: column vector, the best point found
%    f: the value of the function at x
%    loc: local optimality certificate, structure with 2 fields:
%      loc.dnorm: norm of a vector in the convex hull of bundled or
%          sampled gradients of the function evaluated at and near x 
%      loc.evaldist: specifies max distance from x at which these gradients 
%          were evaluated.  
%       The smaller loc.dnorm and loc.evaldist are, the more likely 
%       it is that x is an approximate local minimizer.
%       If quadprog is not in path, loc.dnorm is the norm of the gradient 
%       at x and loc.evaldist is 0.
%    Y: columns are points where these gradients were evaluated, including x
%    G: the gradients of the function at the columns of Y
%    w: vector of positive weights summing to one such that dnorm = ||G*w||
%    H: the final BFGS inverse Hessian approximation, typically 
%        very ill-conditioned if the function is nonsmooth
%
%   Method: 
%      BFGS phases: BFGS is run from options.phasenum(1) different starting 
%       points. The starting points are taken from the columns of options.x0
%       if provided, and otherwise are generated randomly. Each BFGS phase
%       runs until the line search fails to return a better point, or
%       options.phasemaxit(1) iterations are reached, or the norm of the 
%       gradient <= options.normtol. HANSO terminates if the gradient norm
%       test was satisfied at the lowest point found or QUADPROG is not in
%       path, otherwise it continues to:
%      Local bundle phases:  up to options.phasenum(2) local bundle phases 
%       are run from the lowest point found by BFGS, in an attempt to
%       verify local optimality at this point. Each local bundle phase 
%       keeps adding gradients encountered in the line search, terminating 
%       if the step is more than options.evaldist, or options.phasemaxit(2) 
%       iterations are reached, or the norm of the smallest vector in the 
%       convex hull of the bundled gradients <= options.normtol. 
%       If options.phasenum(2) > 1, BFGS is run between each unsuccessful
%       local bundle phase and the next attempt, terminating as before, 
%       but with a limit of options.phasemaxit(2) iterations.
%       HANSO terminates if the local optimality condition was satisfied,
%       otherwise it continues to:
%      Gradient sampling phases: options.phasenum(3) gradient sampling
%       phases are run from lowest point found, with sampling radii from
%       10^(options.phasenum(3)-1)*evaldist down to evaldist. Each gradient 
%       sampling phase is terminated when options.phasemaxit(3) iterations
%       are reached, or the norm of the smallest vector in the convex hull 
%       of the sampled gradients <= options.normtol. Unlike local bundle, 
%       which tries to verify optimality of a point found by BFGS without 
%       moving much, gradient sampling sometimes substantially improves the 
%       approximation found by BFGS, especially if the objective is
%       not Lipschitz or has a big Lipschitz constant. The number of 
%       sampled gradients is controlled by options.ngrad. HANSO terminates
%       at the end of the final gradient sampling phase.
%      Termination takes place immediately during any phase if
%       options.cpumax CPU time is exceeded.
%
%   James V. Burke, Adrian S. Lewis and Michael L. Overton
%   Send comments/bug reports to Michael Overton, overton@cs.nyu.edu,
%   with a subject header containing the string "hanso".

% parameter defaults
if nargin == 0
   error('hanso: "pars" is a required input parameter')
end
if nargin == 1
   options = [];
end
% call setdefaults_hanso first so its defaults take precedence
options = setdefaults_hanso(pars, options); % special fields for HANSO
options = setdefaults(pars, options); % this routine is called by other codes too
options = rmfield(options, 'maxit'); % remove before displaying options below
cpufinish = cputime + options.cpumax;
normtol = options.normtol;
evaldist = options.evaldist;
fvalquit = options.fvalquit;
prtlevel = options.prtlevel;
phasemaxit = options.phasemaxit;
phasenum = options.phasenum;
quadprog = options.quadprog;
if prtlevel > 0  % Version 1.0 July 2006, Version 1.01 May 2009 (minor changes to BFGS only)
    fprintf('HANSO Version 1.01, optimizing objective %s over %d variables with options\n', ...
        pars.fgname, pars.nvar')
    disp(options)
end
if options.quadprog == -1
    phasenum(2) = 0;
    phasenum(3) = 0;
    if prtlevel > 0
        fprintf('hanso: cannot run bundle or gradient sampling phases since QUADPROG is not in path\n')
        fprintf('hanso: *** install optimization toolbox or MOSEK for better results\n')
    end
end
% BFGS Phase, options.cpumax does not need resetting as nothing done yet
options.nstart = phasenum(1); % bfgs will use this to augment options.x0 if necessary
options.maxit = phasemaxit(1); % for bfgs
[x, f, g, H] = bfgs(pars, options); % multiple starting points in general
if length(f) > 1 % more than one starting point
    [f,indx] = min(f); % throw away all but the best result
    x = x(:,indx);
    g = g(:,indx);
    H = H{indx}; % bug if do this when only one start point: H already matrix
end
gnorm = norm(g);
if gnorm < normtol | cputime > cpufinish | f < fvalquit | f == inf |...
        phasenum(2) + phasenum(3) == 0
    if gnorm < normtol & prtlevel > 0
        fprintf('hanso: verified optimality within tolerance in bfgs phase\n')
    elseif phasenum(2) + phasenum(3) == 0 & prtlevel > 0
        fprintf('hanso: no post-bfgs phase, finished\n')
    elseif f == inf & prtlevel > 0
        fprintf('hanso: f is infinite at all starting points, quitting\n')
    end % in other cases, message already printed
    loc.dnorm = gnorm; 
    loc.evaldist = 0;
    Y = x;
    G = g;
    w = 1;
    return
end
% local bundle phases
if phasenum(2) > 0
    options.maxit = phasemaxit(2); 
    bundlenum = 0;
    while bundlenum < phasenum(2)
        bundlenum = bundlenum + 1;
        x0 = x; % for BFGS update below
        grad0 = g; % for BFGS update below
        options.cpumax = cpufinish - cputime;  % time left
        [x, f, g, dnorm, Y, G, w] = localbundle(x, f, g, pars, options);
        % here we don't need to check if f < fvalquit or f == inf
        if dnorm < normtol | cputime > cpufinish 
            if dnorm < normtol & prtlevel > 0
                fprintf('hanso: localbundle %d verified optimality within tolerance using %d gradients\n', bundlenum, size(G,2))
            end
            [loc, Y, G, w] = postprocess(x, g, dnorm, Y, G, w, options, f);
            return
        end
        if bundlenum < phasenum(2)
            % rerun bfgs before restarting local bundle again
            % usually very advantageous to reuse previous Hessian approximation, 
            % (although in theory, this prevents getting rid of possible
            %  bad approximations, so this is questionable)
            % either way, seems a good idea to perform an update first,
            % using the latest gradient that was returned by a line search
            s = Y(:,size(Y,2)) - x0; % not always the same as x - x0
            y = G(:,size(G,2)) - grad0;
            sty = s'*y;
            if sty > 0 % excludes possible case that sty is nan
                rho = 1/sty;
                rhoHyst = rho*(H*y)*s';                                      
                H = H - rhoHyst' - rhoHyst + rho*s*(y'*rhoHyst) + rho*s*s';   
                H = 0.5*(H + H'); 
            else
                if prtlevel > 0
                    fprintf('skipping BFGS update prior to local bundle call, sty = %g\n', sty)
                end
            end
            options.H0 = H;
            options.x0 = x;
            options.cpumax = cpufinish - cputime;  % time left
            options.nstart = 1; % no additional random starts!
            [x, f, g, H] = bfgs(pars, options);
            gnorm = norm(g);
            if gnorm < normtol | cputime > cpufinish
                if gnorm < normtol & prtlevel > 0
                    fprintf('hanso: verified optimality within tolerance in bfgs run following local bundle %d\n',...
                        bundlenum)
                end
                loc.dnorm = gnorm;
                loc.evaldist = 0;
                Y = x;
                G = g;
                w = 1;
                return
            end
        end
    end
end
if phasenum(3) == 0 % no gradient sampling, finish
    [loc, Y, G, w] = postprocess(x, g, dnorm, Y, G, w, options, f);
else % launch gradient sampling
    options.maxit = options.phasemaxit(3);
    % set up sampling radii to decrease by factors of 10, ending with evaldist
    samprad = evaldist; 
    for j=1:phasenum(3)-1
        samprad = [10*samprad(1), samprad];
    end
    options.samprad = samprad;
    options.x0 = x;
    options.nstart = 1; % otherwise gradient sampling will augment with random starts
    options.cpumax = cpufinish - cputime;  % time left
    [x, f, g, dnorm, Y, G, w] = gradsamp(pars, options);
    if dnorm < normtol & prtlevel > 0
        fprintf('hanso: gradient sampling verified optimality within tolerance\n')
    end
    [loc, Y, G, w] = postprocess(x, g, dnorm, Y, G, w, options, f);
end