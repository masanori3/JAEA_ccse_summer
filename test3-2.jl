num = 47
numt = 19
numtrain = num*num
numtest = numt*numt
xtrain = range(-2,2,length=num)
ytrain = range(-2,2,length=num)
xtest = range(-2,2,length=numt)
ytest = range(-2,2,length=numt)

count = 0
ztrain = Float32[]
for i = 1:num
    for j=1:num
        count += 1
        push!(ztrain, f(xtrain[i],ytrain[j]))
    end
end

count = 0
ztest = Float32[]
for i = 1:numt
    for j=1:numt
        count += 1
        push!(ztest, f(xtest[i],ytest[j]))
    end
end

inputdata_train = []
count = 0
for i=1:num
    for j=1:num
        count += 1
        push!(inputdata_train,([xtrain[i],ytrain[j]],ztrain[count]))
    end
end

inputdata_test = []
count = 0
for i=1:numt
    for j=1:numt
        count += 1
        push!(inputdata_test,([xtest[i],ytest[j]],ztest[count]))
    end
end

function make_random_batch(data_input,batchsize)
    numofdata = length(data_input)
    A = rand(1:numofdata,batchsize) #インデックスをシャッフル
    data = []
    for i=1:batchsize
        push!(data,data_input[A[i]]) #ランダムバッチを作成。 
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
        Flux.train!(loss, params(model),data, opt)
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
znn =[model([i,j])[1] for i in x, j in y]'
p = plot(x,y,[znn], st=:wireframe)
savefig("dense.png")
