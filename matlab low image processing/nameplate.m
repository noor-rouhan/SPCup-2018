close all; clear all; clc;
i = imread('nameplate1.jpg');
b = rgb2gray(i);
% imshow(b)
e= edge(b,'Canny',.1);
ec= imcomplement(e);
% roi = [40 385 82 380];
results = ocr(i)
% Iocr = insertText(i,ocrResults.Text,'AnchorPoint',...
%     'RightTop','FontSize',16);
% figure; imshow(Iocr);
% word = ocrResults.Words{1:9}
% 
% % Location of the word in I
% wordBBox = ocrResults.WordBoundingBoxes(2,:)
% figure;
% Iname = insertObjectAnnotation(i, 'rectangle', wordBBox, word);
% imshow(Iname);
% The regular expression, '\d', matches the location of any digit in the
% recognized text and ignores all non-digit characters.
regularExpr = '\d';

% Get bounding boxes around text that matches the regular expression
bboxes = locateText(results, regularExpr, 'UseRegexp', true);

digits = regexp(results.Text, regularExpr, 'match');

% draw boxes around the digits
Idigits = insertObjectAnnotation(i, 'rectangle', bboxes, digits);

figure;
imshow(Idigits);