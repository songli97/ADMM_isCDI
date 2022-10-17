function FL = make_laplacian(n)
%%make Laplacian 
Lap = zeros(n,n);
Lap(1,1) = 4;
Lap(1,2) = -1;
Lap(n,1) = -1;
Lap(2,1) = -1;
Lap(1,n) = -1;
FL = fft2(Lap);
end
