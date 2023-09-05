using Plots
# read kadai2d_epoch_loss_function.txt
opt_name = ["Adam", "Descent", "Nesterov", "AdaGrad"]
for j = 1:length(opt_name)
    filename = "kadai2d_epoch_loss_function_" * opt_name[j] * ".txt"
    dataset = readlines(filename)
    numdata = countlines(filename)

    epoch = zeros(numdata)
    loss = zeros(numdata)
    
    for i = 1:length(dataset)
        spdata = split(dataset[i])
        for ii = 1:2
            if ii == 1
                epoch[i] = parse(Float64,spdata[ii])
            end
            if ii == 2
                loss[i] = parse(Float64,spdata[ii])
            end

        end
    end
    #transform to log
    log10_loss = log10.(loss)    
    #plot
    plot!(epoch, log10_loss, xlabel="epoch", ylabel="log10_loss", label=opt_name[j])
    savefig("eposh_loss_2d_comparison.png")
end


