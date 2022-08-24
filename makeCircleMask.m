function mask = makeCircleMask(r,width, y,x)


mask = zeros(width,width);
for i=1:width
    for j=1:width
        if (i-x)^2+(j-y)^2 < r^2
            mask(j,i) = 1;
        end
    end
end
%[x1,y1] = meshgrid(1:width,1:width);
%mask = (((x1-x).^2 + (y1-y).^2) < r^2);