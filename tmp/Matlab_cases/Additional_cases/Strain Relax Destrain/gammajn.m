function y=gammajn(j,n,RL)

yt=RL-RL;
for i=1:(n+j)
    yt=yt+RL.^(i-1)./factorial(i-1);
end
y=1-exp(-RL).*yt;