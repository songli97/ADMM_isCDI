clear;
clc;
tic
load('exp_data_glioblastoma.mat');
load('R_phs.mat');
m = size(diff_pats,1);

x1 = 530; x2 = 760; y1 = 691; y2 = 921;n1 = 231;n2 = 231;
%n1 = 231;n2 = 231;
O = ones(m,m);

cen = floor(size(diff_pats)./2)+1;
params.ds_cen = [645, 806];%[cen(1)+10 cen(2)+125];
params.size_crop = 140;
ref_mask = single(ref_mask);
%M = supp-ref_mask;
supp = single(supp);
upad = supp;
uhpad = supp;
probe1 = probe(530:760,691:921);
supp1 = supp(530:760,691:921);
params.mask = makeCircleMask(params.size_crop/2, size(diff_pats,1), params.ds_cen(1), params.ds_cen(2));
%%
%ADMM rec
for index = 4:4:48
D = diff_pats(:,:,index);y = D;
rho = 1.0; gamma = 0.01; tau = 1.0; mupad = upad; ppad = upad .*probe; nn = m*m;
u = rand(n1,n2); phi = ones(m,m);uh = zeros(n1,n1);l = -y; mu = zeros(n1,n2);

Lap = make_laplacian(n1);
mask_new = params.mask(x1:x2,y1:y2);

gt = (crop_roi(abs(R(:,:,index).*params.mask), params.size_crop, params.ds_cen(2), params.ds_cen(1)));
for i = 1:30
    u_old = u;
    upad_old = upad;
    %L2-norm
    %u-update
    uh_rhs = rho*nn*(ifft2(ifftshift(y.*phi-l)));
    uh_rhs1 = conj(probe1).*uh_rhs(x1:x2,y1:y2) +tau*(u+mu);
    uh = uh_rhs1./(tau * ones(n1,n2) + rho* nn * probe1 .* conj(probe1));
    %max(max(abs(uh)))
    uhpad(x1:x2,y1:y2) = uh;
    %phi-update
    phi_hat = l+fftshift(fft2(uhpad.*probe));
    mask = (y==0.0);
    phi = zeros(m,m)+ mask + (1-mask).*phi_hat./(y+mask);
    phi = phi ./ abs(phi);    
%     u-update
%   u = tau/(gamma+tau)*(uh-mu);
    urhs = fft2(tau*(uh-mu));
    u = ifft2(urhs./(tau+gamma*(Lap)));
%     u = u/(max(max(abs(u))));
    upad(x1:x2,y1:y2) = u;
    um = upad .* params.mask;
%     urhs = fft2(um);
%     um = ifft2(urhs./(1+1.0*(Lap)));
    upad = upad .* (1-params.mask) + um;
    u = upad(x1:x2,y1:y2);
    %dual variabe update
    l = l + fftshift(fft2(probe.*upad))-y.*phi;
    history(index,i) = norm(u-u_old)/norm(u);
    %history(index,i) = norm(abs(fftshift(fft2(probe.*upad)))-y);
    mu = mu + u - uh;
    mupad(x1:x2,y1:y2) = mu.*supp1;
    norm(u-u_old)/norm(u)
     if(norm(u-u_old)/norm(u)<0.1&&i>1)
         break
     end
end

ADMM(:,:,index/4) = upad;%./max(max(abs(upad)));
end
toc
%% Display result
for n = 1:size(ADMM, 3)
            subplot(3, 4, n)
            reconstruction = crop_roi(angle(ADMM(:,:,n)).*params.mask, params.size_crop, params.ds_cen(2), params.ds_cen(1));
            imagesc(flipud(reconstruction)); axis image off
end
figure;
for n = 1:size(ADMM, 3)%(1:12)*4%
            %subplot(3, 4, n/4)
            subplot(3, 4, n)
            reconstruction = crop_roi(angle(R(:,:,n*4)).*params.mask, params.size_crop, params.ds_cen(2), params.ds_cen(1));
            imagesc(flipud(reconstruction)); axis image off
end
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