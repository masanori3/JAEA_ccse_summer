

using Random
using Flux
using Plots
function main()
include("readfile_3d.jl")    

traindatasize = Int(numdata * 0.9)
testdatasize = Int(numdata * 0.1)

A = shuffle(1:numdata)
A_train = A[1:traindatasize]
w_train_array = w[A_train]
x_train_array = x[A_train]
y_train_array = y[A_train]
z_train_array = z[A_train]
A_test = A[traindatasize + 1 : numdata]
w_test_array = w[A_test]
x_test_array = x[A_test]
y_test_array = y[A_test]
z_test_array = z[A_test]
inputdata_train = []
inputdata_test = []

count = 0
for i=1:traindatasize
    push!(inputdata_train,([w_train_array[i],x_train_array[i],y_train_array[i]],z_train_array[i]))
    
end 

count = 0
for j=1:testdatasize
    push!(inputdata_test,([w_test_array[i],x_test_array[j],y_test_array[j]],z_test_array[j]))
    
end

#= push!(inputdata_train,([x[A_train], y[A_train]], z[A_train]))
push!(inputdata_test,([x[A_test], y[A_test]], z[A_test])) =#

function make_random_batch(data_input,batchsize)
    numofdata = length(data_input)
    D = rand(1:numofdata,batchsize) #インデックスをシャッフル
    data = []
    for i=1:batchsize
        push!(data,data_input[D[i]]) #ランダムバッチを作成。 
    end
    return data
end

model = Chain(Dense(2,10,relu),Dense(10,10,relu),Dense(10,10,relu),Dense(10,1))
loss(x,y) = Flux.mse(model(x), y)
opt = ADAM() 


function train_batch!(data_train,data_test,model,loss,opt,nt)
    batchsize = 128
    for it=1:nt
        data = make_random_batch(data_train,batchsize)
        Flux.train!(loss, Flux.params(model),data, opt)
        if it% 100 == 0
            lossvalue = 0.0
            #testmode!(model, true)
            for i=1:length(data_test)                
                lossvalue += loss(data_test[i][1],data_test[i][2])
            end
            #testmode!(model, false)
            println("$(it)-th loss = ",lossvalue/length(data_test))
        end
    end
end

nt = 3000
train_batch!(inputdata_train,inputdata_test,model,loss,opt,nt) #学習

znn =[model([w_test_array[i],x_test_array[i],y_test_array[i]])[1] for i=1:testdatasize]
histogram(znn)
savefig(dense_output_3d)
end
main()

