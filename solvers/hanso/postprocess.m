function [loc, Y, G, w] = postprocess(x, g, dnorm, Y, G, w, options, f)
% postprocessing of set of sampled or bundled gradients
% if x is not one of the columns of Y, prepend it to Y and
% g to G and recompute w and dnorm: this can only reduce dnorm
% also set loc.dnorm to dnorm and loc.evaldist to the
% max distance from x to columns of Y
% note: w is needed as input argument for the case that w is not
% recomputed but is just passed back to output
% options and f are needed only for printing

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
for j = 1:size(Y,2)
    dist(j) = norm(x - Y(:,j));
end
evaldist = max(dist); % for returning
[mindist, indx] = min(dist); % for checking if x is a column of Y
if mindist == 0 & indx == 1
    % nothing to do
elseif mindist == 0 & indx > 1
    % swap x and g into first positions of Y and G
    % might be necessary after local bundle
    Y(:,[1 indx]) = Y(:,[indx 1]);
    G(:,[1 indx]) = G(:,[indx 1]);
    w([1 indx]) = w([indx 1]);
else
    % prepend x to Y and g to G and recompute w
    Y = [x Y];
    G = [g G];
    % tell quadprog to keep quiet
    options.qp_options = optimset('Display','off','Diagnostics','off'); 
    [w,d] = qpsubprob(G, options);
    dnorm = norm(d);
end
loc.dnorm = dnorm;
loc.evaldist = evaldist;
if options.prtlevel > 0
    fprintf('hanso: best point found has f = %g with local optimality measure: dnorm = %5.1e, evalidst = %5.1e\n',...
        f, dnorm, evaldist)
end
