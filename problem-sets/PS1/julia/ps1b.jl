using CSV 
using DataFrames
using LinearAlgebra
using Plots 

# train logistic reg with newtons method 

# classification problem 
# use sigmoid fn as hypothesis fn 
# theta has same dim as input vector x 

function train(data_path)

    # hypothesis 
    # h = 1/(1+exp(-1 * theta dotp x))
    function sigmoid(x)
        inv(1.0 + exp(-x))
    end


    data = CSV.read(data_path, DataFrame)


    # seed theta, alpha params here 
    theta = zeros(size(data)[2]) # dim(x) + 1 
    epsilon = 1.0 * 10^-5

    # update rule 
    # theta_j = theta_j + alpha * (y_i - h(x_i)) x_j^i
    norm_theta = 10.0^5
    theta_prev = theta
    epoch = 0
    while norm_theta > epsilon
        # maximize theta using newtons method
        # 
        # theta := theta - H^-1 gradient_theta
        (hessian, grad_theta) = reduce(eachrow(data); init=(zeros(3,3), zeros(3))) do (hessian, grad_theta), row
            x = vcat(Vector(row[1:end-1]), [1])
            y = row[end]
            h_x = sigmoid(transpose(theta) * x)
            hessian += x * transpose(x) .* h_x * (1 - h_x)
            grad_theta += (y .- h_x) .* x
            (hessian, grad_theta)
        end

        theta = theta - inv(hessian) * grad_theta 
        theta_prev = theta
        norm_theta = norm(theta - theta_prev)
        epoch = epoch + 1
        if epoch % 10 == 0
            println(["Training iteration:", theta, norm_theta, epoch])
        end

    end

    function predict(row)
        sigmoid(transpose(theta) * vcat(Vector(row), [1]))
    end

    return predict
end


ds1t = CSV.read("/home/joey/workspace/cs229-2018-autumn/problem-sets/PS1/data/ds1_train.csv", DataFrame)
println("ds1t Data loaded")

predict = train("/home/joey/workspace/cs229-2018-autumn/problem-sets/PS1/data/ds1_train.csv")

predictions = map(x -> predict(x), eachrow(ds1t[:, 1:2]))
data_with_pred = hcat(ds1t, predictions)
println(map(x -> round(x, sigdigits=2), (data_with_pred[:,3] - data_with_pred[:,4])[1:30, :]))

println("confusion score y = 0: ", sum(filter(x -> x[3] == 0, data_with_pred)[:, 4]))
println("confusion score y = 1: ", 
    sum(
        map(y -> 1 - y[4], 
            eachrow(filter(x -> x[3] == 1, data_with_pred)))))

default(show=false)
total_points = size(data_with_pred)[1]
scatter(data_with_pred[1:total_points,1], 
        data_with_pred[1:total_points,2], 
        [   data_with_pred[1:total_points,3], 
            data_with_pred[1:total_points,4],
            data_with_pred[1:total_points,3] - data_with_pred[1:total_points,4]])
savefig("pred.png")

scatter(1:total_points, [   data_with_pred[1:total_points,3], 
                            data_with_pred[1:total_points,4],
                            data_with_pred[1:total_points,3] - data_with_pred[1:total_points,4]])
savefig("pred_2d.png")

# scatter(ds1t[:,1], ds1t[:,2], ds1t[:,3])
# savefig("ps1.png")

