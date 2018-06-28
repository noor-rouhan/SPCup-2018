close all; clear all; clc;
c = 0;
im = imread('(HTC-1-M7)1.jpg');
tic
 img = gpuArray(im(:,:,2));
    for j =2: size(im,2) -1
        A(:,j) = 2*img(:, j) -img(:,j+1) + img(:, j-1);
        
    end
B = gather(A);
toc
mono_mean = mean(gather(B));
mono_fft = fft(mono_mean);
mono_abs = abs(fft(mono_mean));
n = 3000;
m = abs(mono_fft);
p = unwrap(angle(mono_fft));
f = (0:length(mono_fft)-1)*100/length(mono_fft);
subplot(3,1,1)
plot(f,m);
subplot(3,1,2);
plot(f,p*180/pi);
 subplot(3,1,3);
 plot(m);
 %B = arrayfun(gpusecondorder,img);

