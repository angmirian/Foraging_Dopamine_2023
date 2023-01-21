function y=normaliseNaN(b)

for i=1:size(b,2)
    a=b(:,i);
    y(:,i)=(a-mean(a(~isnan(a))))/std(a(~isnan(a)));
end