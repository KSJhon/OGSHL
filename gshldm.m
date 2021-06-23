function [x, xarr]  = gshldm(y, K, lam, Nit, p)
% [x, cost] = gshldm(y, K, lam, Nit)
% Group-Sparse Hyper-Laplacian regularized denoising.
%
% INPUT
%   y - noisy signal
%   K - group size (small positive integer)
%   lam - regularization parameter (lam > 0)
%   Nit - number of iterations
%
% OUTPUT
%   x - denoised signal


% Ivan Selesnick, selesi@poly.edu, 2012
% Modified by LJ, UESTC, 2013
% Modified by KSJon, NENU, 2018
% history
h = ones(K, K);                                     % For convolution
x = y;                                              % Initialization
p2 = p * 2;

op = sqrt(conv2(abs(x).^p2, h, 'same'));
xarr = sum(sum((x - y).^2))/2 + lam * sum(op(:));
if K ~=1
    
    for k = 1:Nit
        r = sqrt(conv2(abs(x).^p2, h, 'same')) + eps; % zero outside the bounds of  x
        %   r =  sqrt(imfilter(abs(x).^2,h));  % slower than conv2
        v = conv2(1./r, h, 'same');
        %     F = 1./(lam*v) + 1;
        %     x = y - y./F;
        x = y./(1 + p * lam * v.*abs(x).^(p2 - 2));
        
        op = sqrt(conv2(abs(x).^p2, h, 'same'));
        xarr = [xarr sum(sum((x - y).^2))/2 + lam * sum(op(:))];
    end
else
    for k = 1:Nit
        r = sqrt(abs(x).^2); % zero outside the bounds of  x
        %   r =  sqrt(imfilter(abs(x).^2,h));  % slower than conv2
        v = 1./r;
        %     F = 1./(lam*v) + 1;
        %     x = y - y./F;
        x = y./(1 + lam * v);
    end
end

function w = newton_w(v, beta, alpha)

% for a general alpha, use Newton-Raphson; more accurate root-finders may
% be substituted here; we are finding the roots of the equation:
% \alpha*|w|^{\alpha - 1} + \beta*(v - w) = 0

iterations = 4;

x = v;

for a=1:iterations
    fd = (alpha)*sign(x).*abs(x).^(alpha-1)+beta.*(x-v);
    fdd = alpha*(alpha-1)*abs(x).^(alpha-2)+beta;
    
    x = x - fd./fdd;
end

q = find(isnan(x));
x(q) = 0;

% check whether the zero solution is the better one
z = beta./2*v.^2;
f   = abs(x).^alpha + beta./2*(x-v).^2;
w = (f<z).*x;

