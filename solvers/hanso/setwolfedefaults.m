function options = setwolfedefaults(options, CG)
%  check Wolfe line search fields for options and set defaults
%  CG is 1 for CG methods, 0 for BFGS methods

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
if isfield(options, 'strongwolfe')
    if options.strongwolfe ~= 0 & options.strongwolfe ~= 1
        error('setwolfedefaults: input "options.strongwolfe" must be 0 or 1')
    end
else
    if CG
        options.strongwolfe = 1;  % needed for convergence analysis
    else
        options.strongwolfe = 0;  % not needed for convergence analysis
        % strong Wolfe is very complicated and is bad for nonsmooth functions
    end
end
if isfield(options, 'wolfe1') 
    if ~isposreal(options.wolfe1)
        error('setwolfedefaults: input "options.wolfe1" must be a positive real scalar')
    end
else
    options.wolfe1 = 1e-4;
end
if isfield(options, 'wolfe2')
    if ~isposreal(options.wolfe2)
        error('setwolfedefaults: input "options.wolfe2" must be a positive real scalar')
    end
elseif CG == 1
    options.wolfe2 = 0.49;  % must be < .5 for CG convergence theory
else
    options.wolfe2 = 0.9;   % must be < 1 for BFGS update to be pos def
end
if options.wolfe1 <= 0 | options.wolfe1 >= options.wolfe2 | options.wolfe2 >= 1
    fprintf('setwolfedefaults: Wolfe line search parameters violate usual requirements')
end
if options.prtlevel > 0
    if options.strongwolfe & ~CG
        fprintf('Strong Wolfe line search selected, but for BFGS or LMBFGS\n')
        fprintf('weak Wolfe may be preferable, especially if f is nonsmooth\n')
    elseif ~options.strongwolfe & CG 
        fprintf('Weak Wolfe line search selected, but for CG\n')
        fprintf('this often fails: use strong Wolfe instead')
    end
end
            