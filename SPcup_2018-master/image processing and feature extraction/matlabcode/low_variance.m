function y = low_variance( i )
new = i(1:(size(i,1)-rem(size(i,1),100)),1:(size(i,2)-rem(size(i,2),100)),2);
mat_new = mat2cell(new, 100.*ones(1,size(new,1)/100),  100.*ones(1,size(new,2)/100));
var_mat_new = ones(size(mat_new,1),size(mat_new,2));
for i=1:size(mat_new,1)
    for j =1: size(mat_new,2)
        var_mat_new(i,j) = (std2(mat_new{i,j}))^2;
    end
end
[r,c] = find(var_mat_new == min(min(var_mat_new)));
y = mat_new{r,c};
end