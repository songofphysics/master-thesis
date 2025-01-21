using QuantumOptics
using ProgressMeter
using CSV
using DataFrames

function compute_single_wigner(point::Float64, 
                             dim::Int, 
                             r::Float64, 
                             x_grid::Vector{Float64}, 
                             p_grid::Vector{Float64})
    # Check for valid dimensions
    dim > 1 || throw(ArgumentError("Dimension must be greater than 1"))
    
    # Create Fock basis
    b = FockBasis(dim-1)  # Creates a basis with states |0⟩ to |dim-1⟩
    
    # Create vacuum state |0⟩ using 1-based indexing
    Ψ = basisstate(b, 1)  # First state (vacuum) is index 1
    
    # Create and apply squeeze operator
    S = squeeze(b, r)
    Ψ_s = S * Ψ
    
    # Create and apply displacement operator
    D = displace(b, point)
    Ψ_d = D * Ψ_s
    
    # Compute Wigner function
    W = wigner(Ψ_d, x_grid, p_grid)
    
    return W
end

function sample_from_wigner_rejection(wigner_matrix::Matrix{Float64}, 
                                    phase_space_bounds::Tuple{Float64,Float64,Float64,Float64},
                                    n_samples::Int=25000)
    n_samples > 0 || throw(ArgumentError("Number of samples must be positive"))
    
    x_min, x_max, p_min, p_max = phase_space_bounds
    x_min < x_max && p_min < p_max || throw(ArgumentError("Invalid phase space bounds"))
    
    n_x, n_p = size(wigner_matrix)
    
    x_grid = range(x_min, x_max, length=n_x)
    p_grid = range(p_min, p_max, length=n_p)
    
    function get_wigner_value(x::Float64, p::Float64)
        x_idx = clamp(searchsortedfirst(x_grid, x), 1, n_x)
        p_idx = clamp(searchsortedfirst(p_grid, p), 1, n_p)
        return max(0.0, wigner_matrix[x_idx, p_idx])
    end
    
    max_val = maximum(abs.(wigner_matrix))
    max_val > 0 || throw(ErrorException("Wigner function is zero everywhere"))
    
    accepted_samples = Vector{Vector{Float64}}()
    sizehint!(accepted_samples, n_samples)
    max_iterations = n_samples * 1000
    n_iterations = 0
    
    while length(accepted_samples) < n_samples && n_iterations < max_iterations
        x_prop = rand() * (x_max - x_min) + x_min
        p_prop = rand() * (p_max - p_min) + p_min
        
        if rand() * max_val < get_wigner_value(x_prop, p_prop)
            push!(accepted_samples, [x_prop, p_prop])
        end
        n_iterations += 1
    end
    
    n_iterations >= max_iterations && @warn "Maximum iterations reached, only got $(length(accepted_samples)) samples"
    
    # Convert to matrix format
    result = zeros(Float64, length(accepted_samples), 2)
    for (i, sample) in enumerate(accepted_samples)
        result[i,:] = sample
    end
    
    return result
end

function compute_and_sample_wigner(points::Vector{Float64}; 
                                 dim::Int=10, 
                                 r::Float64=100.0,
                                 bounds::Float64=7.0,
                                 n_samples::Int=25000,
                                 output_file::String="wigner_samples.csv")
    
    # Input validation
    isempty(points) && throw(ArgumentError("Points vector cannot be empty"))
    bounds > 0 || throw(ArgumentError("Bounds must be positive"))
    dim > 1 || throw(ArgumentError("Dimension must be greater than 1"))
    n_samples > 0 || throw(ArgumentError("Number of samples must be positive"))
    
    # Create phase space grid
    n_grid = 200
    x_grid = range(-bounds, bounds, length=n_grid) |> collect
    p_grid = range(-bounds, bounds, length=n_grid) |> collect
    phase_space_bounds = (-bounds, bounds, -bounds, bounds)
    
    # Initialize arrays for results
    n_points = length(points)
    wigner_array = zeros(Float64, n_points, n_grid, n_grid)
    samples = Vector{Matrix{Float64}}(undef, n_points)
    
    # Progress meter
    p = Progress(n_points, desc="Processing points: ")
    
    # Sequential processing of points
    for i in 1:n_points
        try
            # Compute Wigner function
            wigner_matrix = compute_single_wigner(points[i], dim, r, x_grid, p_grid)
            wigner_array[i,:,:] = wigner_matrix
            
            # Sample from Wigner function
            samples[i] = sample_from_wigner_rejection(
                wigner_matrix,
                phase_space_bounds,
                n_samples
            )
            
            next!(p)
            
        catch e
            @error "Error processing point $(points[i])" exception=(e, catch_backtrace())
            rethrow(e)
        end
    end
    
    # Convert samples to DataFrame and save
    println("Saving samples to CSV...")
    
    df = DataFrame()
    for (i, point) in enumerate(points)
        point_samples = samples[i]
        df_point = DataFrame(
            input_point = fill(point, size(point_samples, 1)),
            x = point_samples[:,1],
            p = point_samples[:,2]
        )
        append!(df, df_point)
    end
    
    CSV.write(output_file, df)
    println("Samples saved to $output_file")
    
    return samples, wigner_array, phase_space_bounds
end

# Example usage:
points = range(-π, π, length=300) |> collect;
samples, wigner_matrices, bounds = compute_and_sample_wigner(points);