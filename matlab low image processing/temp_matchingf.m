close all; clear all;
clc;
onion   = rgb2gray(imread('shapes.png'));
peppers = rgb2gray(imread('temp_box.bmp'));
imshowpair(peppers,onion,'montage');
c = normxcorr2(onion,peppers);
figure, surf(c), shading flat