__precompile__
using Interpolations
include("MaterialParameters.jl")

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
    # tabulated sigma
    sigma_tab::Array{Float64,2};
    # moment values for IC
    StarMAPmoments::Array{Float64,1};
    # settings
    settings::Settings

    # constructor
    function CSD(settings::Settings)
        # read tabulated material parameters
        param = MaterialParameters();
        S_tab = param.S_tab;
        E_tab = param.E_tab;
        E_sigmaTab = param.E_sigmaTab;
        sigma_tab = param.sigma_tab;

        # define constant cross-sections for linesource test
        if settings.problem == "LineSource"
            S_tab = ones(size(S_tab));
            sigma_tab = zeros(size(sigma_tab));
            sigma_tab[:,1] .= 1.0;
        end

        # compute transformed energy for tabulated energies
        nTab = length(E_tab)
        E_transformed = zeros(nTab)
        for i = 2:nTab
            E_transformed[i] = E_transformed[i - 1] + ( E_tab[i] - E_tab[i - 1] ) / 2 * ( 1.0 / S_tab[i] + 1.0 / S_tab[i - 1] );
        end

        # define minimal and maximal energy for computation
        minE = 5e-5+1e-8;
        maxE = settings.eMax;

        # determine bounds of transformed energy grid for computation
        ETab2ETrafo = LinearInterpolation(E_tab, E_transformed; extrapolation_bc=Throw())
        eMaxTrafo = ETab2ETrafo( maxE );
        eMinTrafo = ETab2ETrafo( minE );

        # determine transformed energy Grid for computation
        nEnergies = Integer(ceil(maxE/settings.dE));
        eTrafo = collect(range(eMaxTrafo - eMaxTrafo,eMaxTrafo - eMinTrafo,length = nEnergies));

        # determine corresponding original energy grid at which material parameters will be evaluated
        ETrafo2ETab = LinearInterpolation(E_transformed, E_tab; extrapolation_bc=Throw())
        eGrid = ETrafo2ETab(eMaxTrafo .- eTrafo)

        # compute stopping power for computation
        E2S = LinearInterpolation(E_tab, S_tab; extrapolation_bc=Throw())
        S = E2S(eGrid)

        # compute stopping power at intermediate time points
        dE = eTrafo[2]-eTrafo[1];
        #SMid = E2S(eGrid.+0.5*dE)

        eGridMid = ETrafo2ETab(eMaxTrafo .- (eTrafo[1:(end-1)].+0.5*dE))
        SMid = E2S(eGridMid)

        new(eGrid,eTrafo,S,SMid,E_sigmaTab,sigma_tab,param.StarMAPmoments,settings);
    end
end

function SigmaAtEnergy(obj::CSD, energy::Float64)
    if energy <= 5e-5
        energy = 5e-5+1e-9
    end
    y = zeros(obj.settings.nPN+1)
    for i = 1:(obj.settings.nPN+1)
        # define Sigma mapping for interpolation at moment i
        E2Sigma = LinearInterpolation(obj.E_Tab, obj.sigma_tab[:,i]; extrapolation_bc=Throw())
        y[i] = E2Sigma(energy)
    end
    return y;
end