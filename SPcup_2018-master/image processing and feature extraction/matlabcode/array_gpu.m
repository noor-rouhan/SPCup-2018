clc; clear all; 
images = transpose( dir( '*.jpg' ) );
array = [];
i = 0;
tic
for file = images
    i = i+1
    img = imread(file.name);
    df = gather( diff(gpuArray(img(:,:,2)),2,2));
    filename = sprintf('(MotoX)%d.csv', i);
    csvwrite(filename,df)
end
toc
%csvwrite('csvlist.csv',m)
%cell2mat(struct2cell(YourStructure))
