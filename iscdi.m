%%
%Matlab code for the paper 
%%Li Song and Edmund Y. Lam, 
%%¡°Fast and robust phase retrieval for masked coherent diffractive imaging,¡± 
%%Photonics Research, vol. 10, no. 3, pp. 758¨C768, March 2022.
%%
%Load data
clear;
clc;
load('exp_data_glioblastoma.mat'); % experiment data
load('R_phs.mat'); %results from the baseline method
%%
%Imaging setup and mask construction
m = size(diff_pats,1);
n1 = 231; % Reconstructed image size
n2 = 231;
O = ones(m,m);

cen = floor(size(diff_pats)./2)+1;
params.ds_cen = [645, 806];
params.size_crop = 140;
ref_mask = single(ref_mask);
M = supp-ref_mask;
supp = single(supp);
xpad = ref_mask;
upad = probe.*(ref_mask);
probe1 = probe(530:760,691:921); % (530:760,691:921):location of the mask
supp1 = supp(530:760,691:921);
params.mask = makeCircleMask(params.size_crop/2, size(diff_pats,1), params.ds_cen(1), params.ds_cen(2));
%%
%ADMM rec
tol = 0.25;
for index = 4:4:48
rho = 1.0; gamma = 0.01; tau = 1.0; nn = m*m; %parameter
x = zeros(n1,n2); phi = ones(m,m);l = zeros(m,m); mu = zeros(n1,n2); %init
D = diff_pats(:,:,index);mupad = probe .* xpad; %aux variable
y = (D+1); %since in D, -1 is the smallest, y is the variable o in the paper
for i = 1:300
    %for termination criteria calculation only
    x_old = x;
    xpad_old = xpad;
    %u-update
    u_rhs = rho*nn*((ifft2(ifftshift(y.*phi-l)))) + tau *(probe .* xpad - mupad);
    u = 1/(tau+rho*nn) * u_rhs(530:760,691:921).*supp1;
    upad(530:760,691:921) = u.*supp1;
    %phi-update
    phi_hat = l+fftshift(fft2(upad));
    mask = (y==0.0);
    phi = zeros(m,m)+ mask + (1-mask).*phi_hat./(y+mask);
    phi = phi ./ abs(phi);    
    %x-update
    x_hat = tau*conj(probe1).*(u+mu)./(gamma + tau * probe1 .* conj(probe1));
    x = x_hat;
    xpad(530:760,691:921) = x.*supp1;
    %dual variabe update
    l = l + (fftshift(fft2(upad))-y.*phi);
    mu = mu + u - probe1.* x;
    mupad(530:760,691:921) = mu.*supp1;
    if(norm(x(:)-x_old(:))/norm(x(:))<tol)
        break;
    end
end
ADMM(:,:,index) = xpad;
end
%toc
%% Display result
h1 = figure(1);
for n = 1:12
            subplot(3, 4, n)
            reconstruction = crop_roi(angle(ADMM(:,:,n*4)).*params.mask, params.size_crop, params.ds_cen(2), params.ds_cen(1));
            imagesc(flipud(reconstruction)); axis image off; colormap(hot);
end
sgtitle('Phase retrieval results from the ADMM method');
h2 = figure(2);

for n = 1:12
            subplot(3, 4, n)
            reconstruction = crop_roi(angle(R(:,:,n*4)).*params.mask, params.size_crop, params.ds_cen(2), params.ds_cen(1));
            imagesc(flipud(reconstruction)); axis image off; colormap(hot);
end
sgtitle('Phase retrieval results from the baseline method');

function [cropped_im] = crop_roi(image, crop_size,centrex,centrey)
% Crop an image to a specified crop_size, centered around centrex, centrey
%   Inputs: 
%       image - image to be cropped
%       crop_size - size of cropped image, can specify [y_dim, x_dim]
%       centrex - x center of cropped image position
%       centrey - y center of cropped image position
%   Outputs:
%       cropped_im - cropped image

if length(crop_size) ~= 1
   crop_size_x = crop_size(2);
   crop_size_y = crop_size(1);
else
   crop_size_x = crop_size;
   crop_size_y = crop_size;
end

if nargin<3 || nargin<4
    centrey=floor(length(image)/2)+1;
    centrex=floor(length(image)/2)+1;
end

bigy = size(image,1);
bigx = size(image,2);

ycent = floor(bigy/2)+1;
xcent = floor(bigx/2)+1;

half_crop_size_x = floor(crop_size_x/2);
half_crop_size_y = floor(crop_size_y/2);

if mod(crop_size,2) == 0
    cropped_im = image(centrey - half_crop_size_y:centrey + (half_crop_size_y - 1),...
    centrex - half_crop_size_x:centrex + (half_crop_size_x - 1), :);
else
    cropped_im = image(centrey - half_crop_size_y:centrey + (half_crop_size_y),...
    centrex - half_crop_size_x:centrex + (half_crop_size_x), :);
end
end