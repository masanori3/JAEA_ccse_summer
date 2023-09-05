filename = "test_2000_3d.txt"
dataset = readlines(filename)
numdata = countlines(filename)

w = zeros(numdata)
x = zeros(numdata)
y = zeros(numdata)
z = zeros(numdata)
for i = 1:length(dataset)
    spdata = split(dataset[i])
    for ii = 1:4
        if ii == 1
            w[i] = parse(Float64,spdata[ii])
        end
        if ii == 2
            x[i] = parse(Float64,spdata[ii])
        end
        if ii == 3
            y[i] = parse(Float64,spdata[ii])
        end
        if ii == 4
            z[i] = parse(Float64,spdata[ii])
        end
        
    end
end
