function [ Fft,Ft,m] = demosaic_fourier( i )
Fft = abs(fft(mean(abs(gather( diff(gpuArray(i(:,:,:)),2,2) )))));
Ft = 0:1/(length(Fft)-1):1;
[row, col] = find(Ft>.4 & Ft<.6);
 m = max(Fft(col(1):col(length(col)))); 
end

