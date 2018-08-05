function [x, f, g, dnorm, X, G, w] = gradsamp1run(x0, f0, g0, pars, options);
% repeatedly run gradient sampling minimization, for various sampling radii
% return info only from final sampling radius
% intended to be called by gradsamp only

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
samprad = options.samprad;
cpufinish = cputime + options.cpumax;
for choice = 1:length(samprad)
    options.cpumax = cpufinish - cputime; % time left
    [x, f, g, dnorm, X, G, w, quitall] = ...
        gradsampfixed(x0, f0, g0, samprad(choice), pars, options);
    if quitall % terminate early
        return  
    end
    % get ready for next run, with lower sampling radius
    x0 = x;   % start from where previous one finished,
                           % because this is lowest function value so far
    f0 = f;
    g0 = g;
end
