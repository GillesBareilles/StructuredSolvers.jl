using DelimitedFiles

function plot_numexps_data(optimdata, pbname, M_opt)
    fig = TikzDocument()

    push!(fig, TikzPicture(StructuredSolvers.plot_fvals_iteration(optimdata)))
    push!(fig, TikzPicture(StructuredSolvers.plot_tangentres_iteration(optimdata)))
    push!(fig, TikzPicture(StructuredSolvers.plot_normalres_iteration(optimdata)))
    push!(fig, TikzPicture(StructuredSolvers.plot_structure_iteration(optimdata, M_opt)))
    push!(fig, TikzPicture(StructuredSolvers.plot_step_iteration(optimdata)))

    push!(fig, TikzPicture(StructuredSolvers.plot_fvals_time(optimdata)))
    push!(fig, TikzPicture(StructuredSolvers.plot_tangentres_time(optimdata)))
    push!(fig, TikzPicture(StructuredSolvers.plot_normalres_time(optimdata)))
    push!(fig, TikzPicture(StructuredSolvers.plot_structure_time(optimdata, M_opt)))
    push!(fig, TikzPicture(StructuredSolvers.plot_step_time(optimdata)))

    println("Building output pdf $(pbname) ...")
    PGFPlotsX.pgfsave("numexps_output/$(pbname).pdf", fig)
    PGFPlotsX.pgfsave("numexps_output/$(pbname).tex", fig)
    return fig
end

function build_table(optimdata, pbname; M_opt=nothing, F_opt=nothing)
    #
    ### Indicators
    #
    # :it, :time, :f_x, :g_x, :norm_step, :norm_minsubgradient_tangent, :norm_minsubgradient_normal,
    # :nb_calls_f, :nb_calls_g, :nb_calls_∇f, :nb_calls_proxg, :nb_calls_∇²fh,
    get_it(os) = os.it
    get_tanres(os) = os.norm_minsubgradient_tangent
    get_normres(os) = os.norm_minsubgradient_normal
    get_F(os) = os.f_x + os.g_x

    dispname(optimizer::ProximalGradient) = "Prox. grad."
    dispname(optimizer::PartlySmoothOptimizer{AlternatingUpdateSelector,WholespaceProximalGradient,ManifoldTruncatedNewton}) = "Newton (tn)"
    dispname(optimizer::PartlySmoothOptimizer{AlternatingUpdateSelector,WholespaceProximalGradient,ManNewtonCG}) = "Newton (CG)"
    dispname(optimizer::PartlySmoothOptimizer{ConstantManifoldSelector{T},WholespaceProximalGradient,ManifoldTruncatedNewton}) where T = "ManNewton (tn)"

    function dispname(optimizer)
        @show typeof(optimizer)
        @show optimizer
        @assert false
        return
    end

    ### Build table from optim data
    table_lines = []
    precisions = [1e0, 1e-2, 1e-4, 1e-6, 1e-8]
    push!(table_lines, ["algs.", "it", "tan-res", "norm-res", "F"])
    #"ncalls f", "ncalls df", "ncalls df_h", "time", "nit", "calls proxgrad"

    build_line(optimizer, data, ind) = [dispname(optimizer), get_it(data[ind]), get_tanres(data[ind]), get_normres(data[ind]), get_F(data[ind])]
    build_line(optimizer, data, ::Nothing) = [dispname(optimizer), "", "", "", ""]


    for (optimizer, data) in optimdata
        for precision in precisions
            # ind = findfirst(os->get_tanres(os)<precision, data)
            ind = findfirst(os->get_tanres(os)<precision, data)
            push!(table_lines, build_line(optimizer, data, ind))

        end
    end

    ### Write table to file
    open("numexps_output/$(pbname).txt", "w") do io
        writedlm(io, table_lines, "&")
    end
    # @show table_lines

    return
end
