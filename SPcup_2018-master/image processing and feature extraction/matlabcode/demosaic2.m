

% corner1 = i(1:512,1:512,2);
% corner2 = i(2448-512:2448,size(i,2)-512:size(i,2),2);
sad = transpose( dir( '*.jpg' ) );
i = 0;
tic
for file = sad
       i = i +1
       % disp( file.name 
       FileName = (file.name);
       [ Fft,Ft,m] = demosaic_fourier( imread(FileName) );
       ip6_max_demosaic(i) = [m];
       m
end
toc



