%% demo_OGSHL for image denoising
% If you use this Matlab code, please cite
% Kyongson Jon, Ying Sun, Qixin Li, Jun Liu, Xiaofei Wang and Wensheng Zhu,
% Image restoration using overlapping group sparsity on hyper-Laplacian prior of image gradient,
% Neurocomputing 420 (2021) 57-69
% This demo successfully run on Matlab 9.7.0.1190202 (R2019b).
close all; clear variables; clc
ima_dir = 'test_images';
% blur kernel
% ones(9,9)/81 for average blur
% fspecial('gaussian', [7 7], 4) for Gaussian blur
% psf = ones(9,9)/81;% average blur
psf = fspecial('gaussian', [7 7], 4);% average blur
BSNR = 40; % noise level for blurred image
K = 3; % generating KxK square window for grouping
Nit = 5; % inner loop iteration number
MaxIter = 200; % outer loop
tol = 1e-8; % stopping criterion
p = 0.8; % hyper-Laplacian shape parameter
lam = 0.005; % you can tune this regularization parameter to get the best improvement.
tau = lam; % penalty parameter

img_file = 'camera(256).png'; % test image file, sharp
I = imread(strcat(ima_dir, filesep, img_file)); I = double(I);

% simulate a blurry image, g

H = BlurMatrix(psf, size(I));
g = H * I;
% add noise
stream = RandStream('mt19937ar', 'Seed', 88);% 88 to reproduce the paper result
RandStream.setGlobalStream(stream);
sigma = BSNR2WGNsigma(g, BSNR); %get sigma value from BSNR
Bn = g +  sigma * randn(size(I)); % get degraded image, Bn = H * I + noise
% measure the quality of degraded image
psnr_blur = psnr(Bn, I, 255);
ssim_blur = SSIM(Bn, I);
% main function
tg = tic;
outg = gshl2d(Bn, I, psf, K, lam, Nit, MaxIter, tau, tol, p);
tg = toc(tg);
% measure the quality of degraded image
psnr_recon = psnr(outg.sol, I, 255);
ssim_recon = SSIM(outg.sol, I);

% show results
display(sprintf('(psnr=%.2f,ssim=%.3f)->(psnr=%.2f,ssim=%.3f)', psnr_blur, ssim_blur, psnr_recon, ssim_recon));

figure;imshow(min(max(Bn, 0), 255), []), title(sprintf('noisy image (PSNR = %3.3f dB), SSIM %3.3f)', psnr_blur, ssim_blur));
figure;imshow(outg.sol, []);title(sprintf('OGSHL''s deblurring (PSNR = %3.2f dB, SSIM %3.3f)', psnr_recon, ssim_recon));