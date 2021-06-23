%% demo_OGSHL for image denoising
% If you use this Matlab code, please cite
% Kyongson Jon, Ying Sun, Qixin Li, Jun Liu, Xiaofei Wang and Wensheng Zhu,
% Image restoration using overlapping group sparsity on hyper-Laplacian prior of image gradient,
% Neurocomputing 420 (2021) 57-69
% This demo successfully run on Matlab 9.7.0.1190202 (R2019b).
close all, clear variables; clc
% data initialization
ima_dir = 'test_images';
dir_info = dir(ima_dir);
p = 0.8; % hyper-Laplacian shape parameter
K = 3; % generating K X K square window for grouping
Nit = 5; % inner loops
MaxIter = 200; % outer loops
tol = 1e-5; % stopping criterion: relative diffrence between two iterative results
sigma = 30; % noise level
tau = 5; % penalty parameter
% regularization parameter lambda can be adjusted by the user, e.g.,
% if sigma == 15, then set to lam = 4;
% if sigma == 30, then set to lam = 10;
lam = 10;

img_file = 'peppers(256).png'; % test image file
I = imread(strcat(ima_dir, filesep, img_file)); I = double(I);

% Next two lines are for setting noise
stream = RandStream('mt19937ar', 'Seed', 23); %to reproduce the paper result
RandStream.setGlobalStream(stream);
% simulate a noisy image, Bn
Bn = I +  sigma * randn(size(I));
% measure the quality of degraded image
psnr_noisy = psnr(Bn, I, 255);
ssim_noisy = SSIM(Bn, I);
tg = tic; % start a stopwatch timer 
outg = gshl2denoise(Bn, I, K, lam, Nit, MaxIter, tau, tol, p);
tg = toc(tg); % end stopwatch timer
% measure the quality of denoised image
psnr_recon = psnr(outg.sol, I, 255);
ssim_recon = SSIM(outg.sol, I);

% show results
display(sprintf('(psnr=%.2f,ssim=%.3f)->(psnr=%.2f,ssim=%.3f)', psnr_noisy, ssim_noisy, psnr_recon, ssim_recon));

figure;imshow(min(max(Bn, 0), 255), []), title(sprintf('noisy image (PSNR = %3.3f dB), SSIM %3.3f)', psnr_noisy, ssim_noisy));
figure;imshow(outg.sol, []);title(sprintf('OGSHL''s deblurring (PSNR = %3.2f dB, SSIM %3.3f)', psnr_recon, ssim_recon));
