using Plots
num = 30
x = range(-2,2,length=num)
y = range(-2,2,length=num)
f(x,y) = x*y + cos(3*x)+exp(y/5)*x + tanh(y)*cos(3*y)
z =[f(i,j) for i in x, j in y]'
p = plot(x,y,z, st=:wireframe)
savefig("original.png")
