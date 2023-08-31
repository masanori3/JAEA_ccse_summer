n = 100
x0 = range(-2,length=n,stop=2) #Julia 1.0.0以降はlinspaceではなくこの書き方になった。
a0 = 3.0
a1= 2.0
b0 = 1.0
y0 = zeros(Float32,n)
f(x0) = a0.*x0 .+ a1.*x0.^2 .+ b0 .+ 3*cos.(20*x0)
y0[:] = f.(x0)

function make_φ(x0,n,k)
    φ = zeros(Float32,k,n)
    for i in 1:k
        φ[i,:] = x0.^(i-1)
    end
    return φ
end
k = 4
φ = make_φ(x0,n,k)

# make model
using Flux
model = Dense(k, 1) #モデルの生成。W*x + b : W[1,k],b[1]
# check "W" and " b"
println("W = ",model.weight)
println("b = ",model.bias)

# Define the loss function to be minimized
loss(x, y) = Flux.mse(model(x), y) #loss関数。mseは平均二乗誤差
opt = ADAM() #最適化に使う関数。ここではADAMを使用。

# Create random batches
function make_random_batch(x,y,batchsize)
    numofdata = length(y)
    A = rand(1:numofdata,batchsize) #インデックスをシャッフル
    data = []
    for i=1:batchsize
        push!(data,(x[:,A[i]],y[A[i]])) #ランダムバッチを作成。 [(x1,y1),(x2,y2),...]という形式
    end
    return data
end


# learning
function train_batch!(xtest,ytest,model,loss,opt,nt)
    for it=1:nt
        data = make_random_batch(xtest,ytest,batchsize)
        Flux.train!(loss, Flux.params(model),data, opt)
        if it% 100 == 0
            lossvalue = 0.0
            for i=1:length(ytest)                
                lossvalue += loss(xtest[:,i],ytest[i])
            end
            println("$(it)-th loss = ",lossvalue/length(y0))
        end
    end
end

batchsize = 20
nt = 2000
train_batch!(φ,y0,model,loss,opt,nt) #学習

println(model.weight) #W
println(model.bias) #b

#Wとbを使って予測値を作る。
ye = [model(φ[:,i])[1] for i=1:length(y0)]
#以下はプロット。
using Plots
ENV["PLOTS_TEST"] = "true"
pls = plot(x0,[y0[:],ye[:]],marker=:circle,label=["Data" "Estimation"])
savefig("comparison_Flux.png")
plot(x0,[y0[:],ye[:]],marker=:circle,label=["Data" "Estimation"])
