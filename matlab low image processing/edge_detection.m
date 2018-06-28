close all;
clear all;
clc;
i = imread('27658832_334302630401052_1965109103_n.jpg');
g = rgb2gray(i);
% BW = edge(g,'Canny');
 BW = edge(g,'Canny',.3);
% BW = edge(g,'Canny',threshold,sigma)
% [BW,threshOut] = edge(g,'Canny',___)
% BW = edge(g,'approxcanny')
% BW = edge(g,'approxcanny',.2);
e = imcomplement(BW);
maskedRgbImage = bsxfun(@times, i, cast(e, 'like', i));
imshow(maskedRgbImage);

