close all;
clear all;
clc;
i = imread('kratomleaf_770.jpg');
g = rgb2gray(i);
% BW = edge(g,'Canny');
 BW = edge(g,'Canny',.03);
% BW = edge(g,'Canny',threshold,sigma)
% [BW,threshOut] = edge(g,'Canny',___)
% BW = edge(g,'approxcanny')
% BW = edge(g,'approxcanny',threshold)
grayImage = double(g);
e = imcomplement(BW);
maskedRgbImage = bsxfun(@times, i, cast(e, 'like', i));
imshow(maskedRgbImage);

