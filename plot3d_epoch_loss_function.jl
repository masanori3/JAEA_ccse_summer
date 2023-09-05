using Plots
# read kadai2d_epoch_loss_function.txt
filename = "kadai3d_epoch_loss_function.txt"
dataset = readlines(filename)
numdata = countlines(filename)

epoch = zeros(numdata)
loss = zeros(numdata)

for i = 1:length(dataset)
    spdata = split(dataset[i])
    for ii = 1:3
        if ii == 1
            epoch[i] = parse(Float64,spdata[ii])
        end
        if ii == 2
            loss[i] = parse(Float64,spdata[ii])
        end
   
    end
end

#transform to log
log_loss = log.(loss)

#plot
plot(epoch, log_loss, xlabel="epoch", ylabel="log_loss")
savefig("eposh_loss_3d.png")