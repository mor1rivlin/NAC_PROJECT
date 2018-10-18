X = -10:0.3:10;
y=repmat(tanh(X),length(X),1).*transpose(repmat(sigmf(X,[1 0]),length(X),1));
[a,b]=meshgrid(X);
figure;
surf(a,b,y)
zlabel('tanh(x) \cdot \sigma (x)')