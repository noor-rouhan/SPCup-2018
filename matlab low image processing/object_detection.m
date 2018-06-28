close all;
clear all;
clc;
i = imread('circles.png');
gray_image = rgb2gray(i);
% imshow(gray_image);
% [centers, radii] = imfindcircles(i,[66  69])
e = edge(gray_image,'Canny',.02);
ec = imcomplement(e);
comp=[];
load templates
for n=1:4 %4->Number of shapes the database TEMPLATES
    sem=corr2(templates{1,n},ec);
    comp=[comp sem];
end

comp=abs(comp);
vd=find(comp==max(comp));
if     vd==1
    letter='TRIANGLE';   
elseif vd==2
    letter='STAR';
elseif vd==3
    letter='CIRCLE';
else
    letter='RECTANGLE';
end
   


