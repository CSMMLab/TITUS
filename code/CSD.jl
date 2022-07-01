__precompile__
using Interpolations
include("MaterialParameters.jl")
include("MaterialParametersProtons.jl")

struct CSD
    # energy grid
    eGrid::Array{Float64,1};
    # transformed energy grid
    eTrafo::Array{Float64,1};
    # stopping power for computational energy grid
    S::Array{Float64,1};
    SMid::Array{Float64,1};
    # tabulated energy for sigma
    E_Tab::Array{Float64,1};
    E_sigmaTab::Array{Float64,1};
    # tabulated sigma
    sigma_tab::Array{Float64,2};
    sigma_pp::Array{Float64,2};
    sigma_ce::Array{Float64,2};
    # moment values for IC
    StarMAPmoments::Array{Float64,1};
    # settings
    settings::Settings

    # constructor
    function CSD(settings::Settings)
        # read tabulated material parameters
        if settings.particle =="Protons"
            param = MaterialParametersProtons();
            S_tab = param.S_tab;
            E_tab = param.E_tab;
            E_sigmaTab = param.E_sigmaTab;
            sigma_tab = param.sigmaEl_OInt_ICRU;
            sigma_pp = param.sigmaEl_tab_pp
            sigma_ce = param.sigma_ce;
        else
            param = MaterialParameters();
            S_tab = param.S_tab;
            E_tab = param.E_tab;
            E_sigmaTab = param.E_sigmaTab;
            sigma_tab = param.sigma_tab;
            sigma_pp = 0 .* param.sigma_tab;
            sigma_ce = 0 .* param.sigma_tab;
        end

        nTab = length(E_tab)
        E_transformed = zeros(nTab)
        for i = 2:nTab
            E_transformed[i] = E_transformed[i - 1] + ( E_tab[i] - E_tab[i - 1] ) / 2 * ( 1.0 / S_tab[i] + 1.0 / S_tab[i - 1] );
        end

        # define minimal and maximal energy for computation
        if settings.particle =="Protons"
            minE = 0.001 .+ settings.eRest;
            maxE = settings.eMax;
        else
            minE = 0.001; # 5e-5+1e-8;
            maxE = settings.eMax;
        end

        # determine bounds of transformed energy grid for computation
        ETab2ETrafo = LinearInterpolation(E_tab, E_transformed; extrapolation_bc=Throw())
        eMaxTrafo = ETab2ETrafo( maxE );
        eMinTrafo = ETab2ETrafo( minE );

        # determine transformed energy Grid for computation
        nEnergies = Integer(ceil(maxE/settings.dE));
        #eTrafo = collect(exp10.(range(log10(eMaxTrafo - eMaxTrafo+0.0001),log10(eMaxTrafo - eMinTrafo -0.0001),length = nEnergies)));
        eTrafo = collect(range(eMaxTrafo - eMaxTrafo,eMaxTrafo - eMinTrafo,length = nEnergies));
        #println("eTrafo", eTrafo)
        # determine corresponding original energy grid at which material parameters will be evaluated
        ETrafo2ETab = LinearInterpolation(E_transformed, E_tab; extrapolation_bc=Throw())
       
        eGrid = ETrafo2ETab(eMaxTrafo .- eTrafo)

        # compute stopping power for computation
        E2S = LinearInterpolation(E_tab, S_tab; extrapolation_bc=Throw())
        S = E2S(eGrid)

        # compute stopping power at intermediate time points
        dE = zeros(length(eTrafo)-1)
        for i=1:length(eTrafo)-1
            dE[i] = eTrafo[i+1]-eTrafo[i];
        end
        #SMid = E2S(eGrid.+0.5*dE)

        eGridMid = ETrafo2ETab(eMaxTrafo .- (eTrafo[1:(end-1)].+0.5.*dE))
        SMid = E2S(eGridMid)
        if settings.particle =="Electrons"
            E_tab = E_sigmaTab;
        end
        new(eGrid,eTrafo,S,SMid,E_tab,E_sigmaTab,sigma_tab,sigma_pp,sigma_ce,param.StarMAPmoments,settings);
    end
end

function SigmaAtEnergy(obj::CSD, energy::Float64)
 if obj.settings.particle =="Protons"
    if energy <= 0.001 .+ obj.settings.eRest
        energy = 0.001 .+ obj.settings.eRest
    end
    y = zeros(obj.settings.nPN+1)
    for i = 1:(obj.settings.nPN+1)
        # define Sigma mapping for interpolation at moment i
            if energy<7 .+ obj.settings.eRest
                E2Sigma_pp = LinearInterpolation(obj.E_Tab, obj.sigma_pp[:,i]; extrapolation_bc=Throw())
                E2Sigma_ce = LinearInterpolation(obj.E_Tab, obj.sigma_ce[:,i]; extrapolation_bc=Throw())
                y[i] = 0.88810600 .* 0 .+ 0.11189400.* E2Sigma_pp(energy) + E2Sigma_ce(energy);
            else
                E2Sigma_O = LinearInterpolation(obj.E_sigmaTab, obj.sigma_tab[:,i]; extrapolation_bc=Throw())
                E2Sigma_pp = LinearInterpolation(obj.E_Tab, obj.sigma_pp[:,i]; extrapolation_bc=Throw())
                E2Sigma_ce = LinearInterpolation(obj.E_Tab, obj.sigma_ce[:,i]; extrapolation_bc=Throw())
                y[i] = 0.88810600 .* E2Sigma_O(energy) .+ 0.11189400.* E2Sigma_pp(energy) + E2Sigma_ce(energy);
            end
    end
    else
    if energy <= 5e-5
        energy = 5e-5+1e-9
    end
    y = zeros(obj.settings.nPN+1)
    for i = 1:(obj.settings.nPN+1)
        E2Sigma = LinearInterpolation(obj.E_Tab, obj.sigma_tab[:,i]; extrapolation_bc=Throw())
        y[i] = E2Sigma(energy)
    end
end
    return y;
end