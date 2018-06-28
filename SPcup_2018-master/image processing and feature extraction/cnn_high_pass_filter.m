close all;  
for l =40:60
l
tic
i = imread(['(iP6)' num2str(l) '.jpg']);
%i1 = imread('(iP6)10.jpg');
%i = imread('(Nex7)1.jpg');
i1 = imread(['(GalaxyS4)' num2str(l) '.jpg']);
I = imread(['(GalaxyS4)' num2str(l+20) '.jpg']);
%I1 = imread('(GalaxyS4)106.jpg');
green = gpuArray(i(1:2000,1:1500,2));
green1 = gpuArray(i1(1:2000,1:1500,2));
green2 = gpuArray(I(1:2000,1:1500,2));
h=(1/12)*[-1 2  -2 2 -1;2 -6 8 -6 2; -2 8 -12 8 -2;2 -6 8 -6 2; -1 2 -2 2 -1] ;
result = imfilter(green,h,'conv');
result1 = imfilter(green1,h,'conv');
result2 = imfilter(green2,h,'conv');                  %iphone .0166 %.0104 %.0110 %.0087 %.0124
cor_different = corr2(result,result1)
cor_same = corr2(result1,result2)
h = figure;
subplot(2,1,1);
plot(normxcorr2(result,result1));
title(['ip6 vs (GalaxyS4). correlation coefficeint: ' num2str(cor_different) ' ']);
subplot(2,1,2);
plot(normxcorr2(result1,result2));
title(['motomax vs (GalaxyS4)1. correlation coefficeint: ' num2str(cor_same) ' ']);
%subplot(4,1,3);
%plot(xcorr2(result,result1));
%title("ip6 vs (GalaxyS4)");
%subplot(4,1,4);
%plot(xcorr2(result1,result2));
%title("htc vs (GalaxyS4)");
saveas(h,sprintf('FIG_ip6_(GalaxyS4)%d.png',l));
toc
end

