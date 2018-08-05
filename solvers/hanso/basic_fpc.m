% function is |x|_1+mu*|Ax-b|^2
% gradient is x/abs(x)+2mu*A'(Ax-b)=x/abs(x)+2uA'Ax-2uA'b
%
% A is the time series matrix without column i, if column i is used for
% prediction. b is column i in A, shifted up by 1.
function [f, g]=basic_fpc(x, pars)
    A=pars.A;
    AA=pars.AA;
    Ab=pars.Ab;
    mu=pars.mu;
    f=norm(x, 1)+mu*norm(A*x-pars.b, 2)^2;
    g=sign(x)+2*mu*(AA*x)-2*mu*Ab;
end

function grad_norm1_x=gradientx(x)
    grad_norm1_x=ones(size(x, 1), 1);
    for i=1:size(grad_norm1_x, 1)
        if x(i)<0 
            grad_norm1_x(i)=-1;
        elseif x(i)==0
            grad_norm1_x(i)=0;
        end
    end
end