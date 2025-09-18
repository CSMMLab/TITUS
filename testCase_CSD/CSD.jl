__precompile__
using Interpolations
include("MaterialParameters.jl")
include("MaterialParametersProtons.jl")

struct CSD{T<:AbstractFloat}
    # energy grid
    eGrid::Array{T,1};
    # transformed energy grid
    eTrafo::Array{T,1};
    # stopping power for computational energy grid
    S::Array{T,2};
    SMid::Array{T,2};
    # tabulated energy for sigma/stopping power
    E_Tab::Array{T,1};
    # tabulated sigma
    sigma_ce::Array{T,3};
    sigma_xi::Array{T,3};
    # settings
    settings::Settings

    # constructor
    function CSD(settings::Settings,T::DataType=Float64)
        # read tabulated material parameters
        if settings.particle =="Protons"
            param = MaterialParametersProtons(settings,settings.OmegaMin);
            S_tab = param.S_tab;
            E_tab = param.E_tab;
            sigma_ce = param.sigma_ce;
            sigma_xi = param.sigma_xi;
        else 
            param = MaterialParameters(settings);
            S_tab = param.S_tab;
            E_tab = param.E_tab;
            sigma_ce = param.sigma_tab;
            sigma_xi = 0 .* sigma_ce; #placeholder until FP implemented for electrons
        end
        nTab = length(E_tab)
        E_transformed = zeros(nTab)
        for i = 2:nTab
            E_transformed[i] = E_transformed[i - 1] + ( E_tab[i] - E_tab[i - 1] ) / 2 * ( 1.0 / S_tab[i] + 1.0 / S_tab[i - 1] );
        end

        # define minimal and maximal energy for computation
        minE = settings.eMin .+ settings.eRest;
        maxE = settings.eMax;

        eTrafoMax = integrate(E_tab, 1 ./S_tab[:,1])
        eTrafo1 = zeros(nTab)
        for i = 1:length(E_tab)
            eTrafo1[i] = eTrafoMax  - integrate(E_tab[1:i], 1 ./S_tab[1:i,1])
        end
        
        ETab2ETrafo = LinearInterpolation(E_tab, eTrafo1; extrapolation_bc=Flat())
        eMaxTrafo = ETab2ETrafo( maxE );
        eMinTrafo = ETab2ETrafo( minE );
        nEnergies = Integer(ceil(maxE/settings.dE));
        if ~iseven(nEnergies)
            nEnergies = nEnergies + 1;
        end
        println("number energies = $nEnergies")
        eGrid = collect(range(minE,maxE,length=nEnergies))[end:-1:1]
        #eGrid = collect(exp.(range(log(minE),log(maxE),length=nEnergies)))[end:-1:1]
        dEGrid=zeros(length(eGrid)-1)
        for i=2:length(eGrid)
            dEGrid[i-1] = eGrid[i-1] - eGrid[i]
        end
        #eTrafo = collect(log.(range(exp.(eMaxTrafo),exp.(eMinTrafo),length = nEnergies)));
        #eTrafo = collect(range(eMaxTrafo,eMinTrafo,length = nEnergies));
        ETrafo2ETab = LinearInterpolation(eTrafo1[end:-1:1], E_tab[end:-1:1].-settings.eRest; extrapolation_bc=Flat())
        eTrafo = ETab2ETrafo(eGrid)
        #eGrid = ETrafo2ETab(eTrafo) .+settings.eRest

        # compute stopping power for computation
        if size(S_tab,2) == 1 #only one material/waterequivalent
            S = zeros(size(eGrid,1),1)
            SMid = zeros(size(eGrid,1),1)

            E2S = LinearInterpolation(E_tab, S_tab[:,1]; extrapolation_bc=Flat())
            S[:,1] = E2S(eGrid)
            # compute stopping power at intermediate time points
            dE = zeros(length(eTrafo)-1)
            for i=1:length(eTrafo)-1
                dE[i] = eTrafo[i+1]-eTrafo[i];
            end
            SMid[:,1] = E2S(eGrid[1:end].-0.5*dEGrid[1])

            eGridMid = ETrafo2ETab(eMaxTrafo .- (eTrafo[1:(end-1)].+0.5.*dE))
        else
            nPsi = size(S_tab,2)
            E2S = interpolate((E_tab[1:end],1:nPsi), S_tab,(Gridded(Linear()),NoInterp()))
            S=zeros(nEnergies,nPsi)
            for i = 1:nEnergies
                S[i,:] = E2S.(eGrid[i],1:nPsi)
            end
            # compute stopping power at intermediate time points
            dE = zeros(length(eTrafo)-1)
            for i=1:length(eTrafo)-1
                dE[i] = eTrafo[i+1]-eTrafo[i];
            end
            SMid=zeros(nEnergies-1,nPsi)
            for i = 1:nEnergies-1
                SMid[i,:] = E2S.(eGrid[i].-0.5*dEGrid[i],1:nPsi)
            end
        end
        new{T}(eGrid,eTrafo,S,SMid,E_tab,sigma_ce,sigma_xi,settings);
    end
end

function XiAtEnergy(obj::CSD{T}, energy::T) where {T<:AbstractFloat}
    if size(obj.sigma_xi,2)==1
        E2Sigma_xi = LinearInterpolation(obj.E_Tab, obj.sigma_xi; extrapolation_bc=Flat())
        y = E2Sigma_xi(energy);
    else
        y = zeros(2,)
        E2Sigma_xi1 = LinearInterpolation(obj.E_Tab, obj.sigma_xi[:,1]; extrapolation_bc=Flat())
        E2Sigma_xi2 = LinearInterpolation(obj.E_Tab, obj.sigma_xi[:,2]; extrapolation_bc=Flat())
        y[1] = E2Sigma_xi1(energy)
        y[2] = E2Sigma_xi2(energy)
    end
end

function XiAtEnergyandX(obj::CSD{T}, energy::T) where {T<:AbstractFloat}
    nPsi = size(obj.sigma_xi,2)
    y = zeros(2,nPsi)
    E2Sigma_xi1 = interpolate((obj.E_Tab,1:nPsi), obj.sigma_xi[:,:,1],(Gridded(Linear()),NoInterp()))
    E2Sigma_xi2 = interpolate((obj.E_Tab,1:nPsi), obj.sigma_xi[:,:,2],(Gridded(Linear()),NoInterp()))
    y[1,:] = E2Sigma_xi1.(energy,1:nPsi)
    y[2,:] = E2Sigma_xi2.(energy,1:nPsi)
    return T.(y);
end

function SigmaAtEnergy(obj::CSD{T}, energy::T) where {T<:AbstractFloat}
    y = zeros(obj.settings.nPN+1)
    for i = 1:(obj.settings.nPN+1)
        # define Sigma mapping for interpolation at moment i
        E2Sigma_ce = LinearInterpolation(obj.E_Tab, obj.sigma_ce[:,i]; extrapolation_bc=Flat())
        y[i] = E2Sigma_ce(energy);
    end
    return T.(y);
end

function SigmaAtEnergyandX(obj::CSD{T}, energy::T) where {T<:AbstractFloat}
    nPsi = size(obj.sigma_ce,3)
    y = zeros(obj.settings.nPN+1,nPsi)
    for i = 1:(obj.settings.nPN+1)
        # define Sigma mapping for interpolation at moment i
        E2Sigma_ce = interpolate((obj.E_Tab,1:nPsi), obj.sigma_ce[:,i,:],(Gridded(Linear()),NoInterp()))
        y[i,:] = E2Sigma_ce.(energy,1:nPsi);
    end
    return T.(y);
end
