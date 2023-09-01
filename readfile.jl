filename = "test_5002d.txt"
dataset = readlines(filename)
numdata = countlines(filename)

x = zeros(numdata)
y = zeros(numdata)
z = zeros(numdata)
for i = 1:length(dataset)
    spdata = split(dataset[i])
    for ii = 1:3
        if ii == 1
            x[i] = parse(Float64,spdata[ii])
        end
        if ii == 2
            y[i] = parse(Float64,spdata[ii])
        end
        if ii == 3
            z[i] = parse(Float64,spdata[ii])
        end
        
    end
end

A = rand(1:numofdata,batchsize)