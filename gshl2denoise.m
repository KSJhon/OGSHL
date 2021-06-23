function out = gshl2denoise(f, I, K, lam, Nit, MaxIter, tau, tol, p)
% GSHL2D for 2D image denoising
%   f:         observed noisy image
%   K:         window size for grouping
%   lam:       regularization parameter
%   Nit:       maximum iterations for inner loop
%   MaxIter:   maximum iterations for outer loop
%   tau:       penalty parameter of ADMM
%
[m, n]= size(f);
tau3 = tau;
DE = diagE(m, n);
d1 = zeros(m, n);
d2 = d1;
d3 = d1;
const = 1;
k = 0;
u = f;
% initialization
p2 = p * 2;
tempfunc = 0;
% following two lines are for dynamic demonstration
[Dux, Duy] = ForwardD(u);

while const
    k = k + 1;
    % ==================
    %     V-subprolem
    % ==================
    % solving v-subproblem
    [vx, ~] = gshldm(Dux - d1, K, lam/tau, Nit, p);
    [vy, ~] = gshldm(Duy - d2, K, lam/tau, Nit, p);
    % ==================
    %     Z-subprolem
    % ==================
    z = min(255, max(u + d3, 0));
    % ==================
    %     U-subprolem
    % ==================
    % solving u-subproblem
    temp = f + tau * div(vx + d1, vy + d2) + tau3 * (z - d3);
    u = fft2(temp)./(tau * DE + 1 + tau3);
    % next two lines are for inpainting, but do not work.
    u = real(ifft2(u));
    
    % ==================
    %  update Lagrangian multipliers d
    % ==================
    d1 = d1 + (vx - Dux);
    d2 = d2 + (vy - Duy);
    d3 = d3 + (u - z);
    
    [Dux, Duy] = ForwardD(u);
    
    gs1 = sqrt(conv2(abs(Dux).^p2, ones(K), 'same'));
    gs2 = sqrt(conv2(abs(Duy).^p2, ones(K), 'same'));
    funcvalue(k) = 0.5 * norm(u - f, 'fro')^2 + lam * sum(gs1(:) + gs2(:));
    
    if k > MaxIter || abs(funcvalue(k) - tempfunc) / abs(funcvalue(k)) < tol
        const = 0;
    end
    tempfunc = funcvalue(k);
end

out.sol = u;
out.funcvalue = funcvalue(1:k);
%% subfunctions
function [Dux,Duy] = ForwardD(u)
Dux = [diff(u, 1, 2), u(:, 1, :) - u(:, end, :)];
Duy = [diff(u, 1, 1); u(1, :, :) - u(end, :, :)];
function DE = diagE(m, n)
DE = abs(psf2otf([1, -1], [m n])).^2 + abs(psf2otf([1; -1], [m n])).^2;

function DtXY = div(X, Y)
% Transpose of the forward finite difference operator
% div = grad^*
DtXY = [X(:, end) - X(:, 1), -diff(X, 1, 2)];
DtXY = DtXY + [Y(end, :) - Y(1, :); -diff(Y, 1, 1)];
