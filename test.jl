using Plots
n = 100
x0 = range(-2,length=n,stop=2) #Julia 1.0.0以降はlinspaceではなくこの書き方になった。
a0 = 3.0
a1= 2.0
b0 = 1.0
y0 = zeros(Float32,n)
f(x0) = a0.*x0 .+ a1.*x0.^2 .+ b0 .+ 3*cos.(20*x0)
y0[:] = f.(x0)
