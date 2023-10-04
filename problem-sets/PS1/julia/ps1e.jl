import Pkg
Pkg.add(["CSV", "DataFrames", "LinearAlgebra", "Plots"])

using CSV 
using DataFrames
using LinearAlgebra
using Plots 


function train(data_path)
    data = CSV.read(data_path, DataFrame)

    x = Matrix(data[:, 1:end-1])
    y = data[:, end]
    model_theta = 1/size(data)[1] * sum(y)
    mu_0 = transpose(x) * (ones(size(y)[1]) - y) / sum(ones(size(y)[1]) - y)
    mu_1 = transpose(x) * y  / sum(y)

    # filter out y=1 and y=0 x elements using row-wise multiplication
    x_yeq1_sub_mu_1 = (x .- transpose(mu_1)) .* y
    x_yeq0_sub_mu_0 = (x .- transpose(mu_0)) .* (ones(size(y)[1]) - y)

    sigma = 1/size(data)[1] * (
        transpose(x_yeq1_sub_mu_1) * (x_yeq1_sub_mu_1) +
        transpose(x_yeq0_sub_mu_0) * (x_yeq0_sub_mu_0)
    )

    theta = 2 * inv(sigma) * mu_1 - 2 * inv(sigma) * mu_0
    theta_0 = transpose(mu_0) * inv(sigma) * mu_0 - transpose(mu_1) * inv(sigma) * mu_1 + log((1 - model_theta) / model_theta)

    function predict(row)
        pyeq1_x = 1 / (1 + (exp( 
            (-1/2 * transpose(row - mu_0) * inv(sigma) * (row - mu_0)) -
            (-1/2 * transpose(row - mu_1) * inv(sigma) * (row - mu_1)) 
            ) * (1 - model_theta) / model_theta))
        if pyeq1_x > 0.5
            1
        else
            0
        end
    end
    return predict 
end 


# train using gaussian discriminate analysis
ds1t = CSV.read("/home/joey/workspace/cs229-2018-autumn/problem-sets/PS1/data/ds1_train.csv", DataFrame)
println("ds1t Data loaded")

predict = train("/home/joey/workspace/cs229-2018-autumn/problem-sets/PS1/data/ds1_train.csv")

data_with_pred = hcat(ds1t, map(predict, eachrow(Matrix(ds1t[:, 1:end-1]))))

error = sum(map(abs, data_with_pred[:, 3] - data_with_pred[:, 4]))
println("error pct ", error/size(ds1t)[1])

