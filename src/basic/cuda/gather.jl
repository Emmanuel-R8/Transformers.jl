"getting vector at the given indices"
gather(w::AbstractMatrix{T}, xs::OneHotArray) where T = gather(w, onehot2indices(xs))

# cpu gather
function gather(w::AbstractArray{T}, xs) where T
    ys = similar(w, size(w, 1), size(xs)...)

    Threads.@threads for i = 1:length(xs)
        ind = Tuple(CartesianIndices(xs)[i])
        @inbounds ys[:, ind...] .= w[:, xs[i]]
    end
    return ys
end

# gpu gather
gather(w::CuMatrix{T}, xs::OneHotArray) where T = gather(w, onehot2indices(xs))
function gather(w::CuMatrix{T}, xs) where T
    ys = CuArray{T}(undef, size(w, 1), size(xs)...)

    function kernel!(ys::CuDeviceArray{T}, w::CuDeviceArray{T}, xs)
        li = threadIdx().y + (blockIdx().y - 1) * blockDim().y
        i = threadIdx().x + (blockIdx().x - 1) * blockDim().x

        if li <= length(xs)
            ind = Tuple(CartesianIndices(xs)[li])
            ys[i, ind...] = w[i, xs[li]]
        end

        return
    end

    max_threads = 256
    threads_x = min(max_threads, size(ys,1))
    threads_y = min(max_threads ÷ threads_x, length(xs))
    threads = (threads_x, threads_y)
    blocks = ceil.(Int, (size(ys,1), length(xs)) ./ threads)

    @cuda blocks=blocks threads=threads kernel!(ys, w, xs)
    return ys
end

# cpu gather back
function ∇gather(Δ::AbstractArray{T}, w::AbstractMatrix{T}, xs) where T
    ys = fill!(similar(w), zero(T))

    Threads.@threads for i = 1:length(xs)
        ind = Tuple(CartesianIndices(xs)[i])
        @inbounds ys[:, xs[i]] .+= Δ[:, ind...]
    end

    return ys
end

#gpu gather back
function ∇gather(Δ::CuArray{T}, w::CuMatrix{T}, xs) where T
    ys = fill!(similar(w), zero(T))

    # Not atomic to do this
    # function kernel!(ys::CuDeviceArray{T}, Δ::CuDeviceArray{T}, xs)
    #     li = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    #     i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    #     if li <= length(xs)
    #         ind = Tuple(CartesianIndices(xs)[li])
    #         ys[i, xs[li]] += Δ[i, ind...]
    #     end

    #     return
    # end

    # max_threads = 256
    # threads_x = min(max_threads, size(ys,1))
    # threads_y = min(max_threads ÷ threads_x, length(xs))
    # threads = (threads_x, threads_y)
    # blocks = ceil.(Int, (size(ys,1), length(xs)) ./ threads)
    # @cuda blocks=blocks threads=threads kernel!(ys, Δ, xs)


    function kernel!(ys::CuDeviceArray{T}, Δ::CuDeviceArray{T}, xs)
        xi = threadIdx().x + (blockIdx().x - 1) * blockDim().x

        if xi <= size(ys, 1)
            for i = 1:length(xs)
                ind = Tuple(CartesianIndices(xs)[i])
                ys[xi, xs[i]] += Δ[xi, ind...]
            end
        end

        return
    end

    max_threads = 256
    threads = max_threads
    blocks = ceil(Int, size(ys, 1) / threads)

    @cuda blocks=blocks threads=threads kernel!(ys, Δ, xs)
    return ys
end


using Flux: Tracker, TrackedArray, TrackedMatrix, data
using Flux.Tracker: @grad, track

gather(w::TrackedMatrix, xs::OneHotArray) = gather(w, onehot2indices(xs))
gather(w::TrackedMatrix, xs) = track(gather, w, xs)

@grad gather(w::TrackedMatrix, xs) = gather(data(w), xs), Δ->(∇gather(Δ, data(w), xs),nothing)