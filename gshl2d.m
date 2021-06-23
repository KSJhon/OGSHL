function out = gshl2d(f, I, psf, K, lam, Nit, MaxIter, tau, tol, p)
% GSHL2D for 2D image denoising
%   f:         observed noisy image
%   K:         window size for grouping
%   lam:       regularization parameter
%   Nit:       maximum iterations for inner loop
%   MaxIter:   maximum iterations for outer loop
%   tau:       penalty parameter of ADMM
[m, n]= size(f);
tau3 = tau;
H = BlurMatrix(psf, [m, n]);
diagHTH = abs(psf2otf(psf, [m, n])).^2;

DE = diagE(m, n);
d1 = zeros(m, n);
d2 = d1;
d3 = d1;
const = 1;
k = 0;
u = f;

% initialization
funcvalue = zeros(MaxIter,1); % function value

tempfunc = 0;
% following two lines are for dynamic demonstration
% imshow(f,[],'InitialMagnification','fit'),title(sprintf('Denoising by OGS method (PSNR = %3.3f dB iteration %.4f) ', psnr_fun(u, I),k))
% pause(1)
[Dux,Duy] = ForwardD(u);
p2 = p*2;
while const
    k=k + 1;
    % ==================
    %     V-subprolem
    % ==================
    
    % solving v-subproblem
    vx = gshldm(Dux - d1, K, lam/tau, Nit, p);
    vy = gshldm(Duy - d2, K, lam/tau, Nit, p);
    % ==================
    %     Z-subprolem
    % ==================
    %     z = u + d3;
    %     z(z>255) = 255;   z(z<0) = 0;
    z = min(255, max(u + d3, 0));
    % ==================
    %     U-subprolem
    % ==================
    % solving u-subproblem
    temp = H' * f + tau*div(vx + d1,vy + d2) + tau3 * (z - d3);
    u = fft2(temp)./(tau*DE + diagHTH + tau3);
    u = real(ifft2(u));
    
    % ==================
    %  Update Lagrangian multipliers d
    % ==================
    d1 = d1 + (vx - Dux);
    d2 = d2 + (vy - Duy);
    d3 = d3 + (u - z);
    % function value
    [Dux, Duy] = ForwardD(u);
    gs1 = sqrt(conv2(abs(Dux).^p2, ones(K), 'same'));
    gs2 = sqrt(conv2(abs(Duy).^p2, ones(K), 'same'));
    funcvalue(k) = 0.5 * norm(H * u - f,'fro')^2 + lam * sum(gs1(:) + gs2(:));
    
    if k > MaxIter || abs(funcvalue(k) - tempfunc) / abs(funcvalue(k)) < tol
        const = 0; %relerr(k)<tol %
    end
    tempfunc = funcvalue(k);
end

out.sol = u;
out.funcvalue = funcvalue(1:k);
%% subfunctions
function [Dux,Duy] = ForwardD(U)
Dux = [diff(U, 1, 2), U(:,1,:) - U(:, end, :)];
Duy = [diff(U, 1, 1); U(1,:,:) - U(end, :, :)];

function DE = diagE(m, n)
DE = abs(psf2otf([1, -1],[m n])).^2 + abs(psf2otf([1; -1],[m n])).^2;

function DtXY = div(X,Y)
% Transpose of the forward finite difference operator
% div = grad^*
DtXY = [X(:, end) - X(:, 1), -diff(X, 1, 2)];
DtXY = DtXY + [Y(end, :) - Y(1, :); -diff(Y, 1, 1)];






