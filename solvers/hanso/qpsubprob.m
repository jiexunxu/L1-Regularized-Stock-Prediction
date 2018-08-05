function [w,d] = qpsubprob(S, options)
%  solve the gradient sampling QP subproblem 
%     min          w'S'Sw = ||S w||^2
%     w 
%
%     subject to   e'w = 1          Sw is a convex combination of gradients
%                   w >= 0          in the columns of S
%  Return w and d = -Sw.
%
%  The QP is solved by a call to quadprog which will invoke either the 
%  Matlab Optimization Toolbox or MOSEK software, depending on which is
%  installed and, if both are, their relative order in the Matlab Path.  
%  Generally, MOSEK is preferable.  To select it, use "addpath" to
%  add it to the front of the path.  To see which is in use, use "which".

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
%
[m,N] = size(S);  % N gradients of length m
H = S'*S;
e = ones(1,N);    % constraint e'w = 1
O = zeros(1,N);
w0 = ones(N,1)/N;  % starting point for quadprog 
% qp_options = optimset('Display','off','TolX', 1e-12, 'TolFun', 1e-12);
% MOSEK occasionally bombs, perhaps if H is not suff pos def?
%%% save qp_debug H O e w0 qp_options
% call to optimset can be very time consuming, do it once in advance
% and pass through options.qp_options
if isfield(options, 'qp_options')
    qp_options = options.qp_options;
else
    qp_options = [];
end
warning_save = warning;  % save the warning state
warning off % optimization toolbox version is incredibly annoying otherwise
% sometimes MOSEK fails if H is not sufficiently positive definite
succeed = 0;
perturb = 1e-16*max(1,norm(H,inf));
count = 1;
while ~succeed
    try
        w = quadprog(H, O, [], [], e, 1, O, [], w0, qp_options);   
        succeed = 1;
    catch
        H = H + perturb*eye(N);
        if options.prtlevel > 0
            fprintf('qpsubprob: quadprog failed, augmenting H with random perturbation\n')
        end
        if count < 10
            perturb = 10*perturb;
            count = count + 1;
        else % should never happen, but just in case
            error('qpsubprog: quadprog failed, even after augmenting H with random perturbations')
        end
    end
end
warning(warning_save); % restore the warning state
if isempty(w)
    error('qpsubprob: quadprog returned empty result: MOSEK license problem? Try smaller limit on number of gradients or switch to Optimization Toolbox')
end
d = -S*w;
% prtlevel = options.prtlevel;
% checktol = 1e-7*max(1, norm(H, inf));
% if prtlevel > 0 & (min(w) < -checktol | abs(sum(w) - 1) > checktol)
%    fprintf('qpsubprob: w computed by quadprog does not satisfy requirements\n')
% end