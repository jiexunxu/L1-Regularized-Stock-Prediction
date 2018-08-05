function options = setdefaults_hanso(pars, options)
% set HANSO defaults that are not set by setdefaults
% called only by hanso
% this is also in setdefaults, but setdefaults_hanso is called first
% so we need it here too, as pars.nvar is referenced below

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
if ~isfield(pars, 'nvar')
   error('hanso: input "pars" must have a field "nvar" (number of variables)')
elseif ~isposint(pars.nvar)
   error('hanso: input "pars.nvar" (number of variables) must be a positive integer')
end
nvar = pars.nvar;
% the following isn't needed since it's in setdefaults, but may as well
% include it so error message says "hanso"
if ~isfield(pars, 'fgname')
   error('hanso: input "pars" must have a field "fgname" (name of m-file computing function and gradient)')
end
if ~isfield(options, 'normtol')
    options.normtol = 1e-4;
elseif ~isposreal(options.normtol)
    error('hanso: input "options.normtol" must be a positive scalar')
end
if ~isfield(options, 'evaldist')
    options.evaldist = 1e-4;
elseif ~isposreal(options.evaldist)
    error('hanso: input "options.evaldist" must be a positive scalar')
end
% options.phasemaxit applies to BFGS, local bundle, gradient sampling
if ~isfield(options, 'phasemaxit')
   % options.phasemaxit = [min([100*nvar, 2000]),...
   %                      nvar + 10,...
   %                       min([10*nvar, 100])];
   options.phasemaxit = [1000, nvar + 10, 100];
elseif size(options.phasemaxit,1)*size(options.phasemaxit,2) ~= 3
    error('hanso: input "options.phasemaxit" must be a vector of length 3')
elseif ~isnonnegint(options.phasemaxit(1))|...
       ~isnonnegint(options.phasemaxit(2))|...
       ~isnonnegint(options.phasemaxit(3))
    error('hanso: entries in input "options.phasemaxit" must be nonnegative integers')
end
% options.phasenum is number of phases in local bundle and gradient sampling
if ~isfield(options, 'phasenum')
    if isfield(options, 'x0')
        if ~isempty(options.x0)
            options.phasenum = [size(options.x0,2) 3 3];
        else
            options.phasenum = [3 3 3];
        end
    else
        options.phasenum = [3 3 3];
    end
elseif size(options.phasenum,1)*size(options.phasenum,2) ~= 3
    error('hanso: input "options.phasenum" must be a vector of length 3')
elseif ~isnonnegint(options.phasenum(1))|...
       ~isnonnegint(options.phasenum(2))|...
       ~isnonnegint(options.phasenum(3))
    error('hanso: entries in input "options.phasenum" must be nonnegative integers')
elseif isfield(options,'x0')
    % number of BFGS phases cannot be less than number of columns of options.x0, if provided
    if options.phasenum(1) < size(options.x0,2)
        options.phasenum(1) = size(options.x0,2);
    end
end
% options.ngrad applies only to grad sampling
if ~isfield(options, 'ngrad')% 150 is the max for the free version of MOSEK
    options.ngrad = min([100, 2*nvar, nvar + 10]);  
elseif ~isposint(options.ngrad)
    error('hanso: input "options.ngrad" must be a positive integer')
end
% options.quadprog tells availability of quadprog
% (may be expensive to keep checking this if called repeatedly)
if ~isfield(options, 'quadprog')
    options.quadprog = 0;  % need to check if it is in path
elseif ~isscalar(options.quadprog)|(options.quadprog ~= 1 & ...
     options.quadprog ~= 0 & options.quadprog ~= -1)
    error('hanso: input "options.quadprog" must be -1, 0 or 1')
end
if options.quadprog == 0
    if exist('quadprog')
        options.quadprog = 1;
    else
        options.quadprog = -1;
    end
end