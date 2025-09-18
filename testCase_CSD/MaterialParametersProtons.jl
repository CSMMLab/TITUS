__precompile__
using Distributions
using LegendrePolynomials
using NumericalIntegration

function Sigma_eModels(E_MeV,mu_0,rho,comp_vector,model)
    #Gives the (macroscopic) elastic scatter cross section based on
    #the energy, material and angle for different models specified by
    #the model variable
    #
    #- E is energy in MeV
    #- mu is cosine of the deflection angle
    #- rho is density
    #- comp_vector is vector with percentage wise composition (according to mass?) of following materials:
    # H_mat  =  1
    # C_mat  =  2
    # N_mat  =  3
    # O_mat  =  4
    # Na_mat =  5
    # Mg_mat =  6
    # P_mat  =  7
    # S_mat  =  8
    # Cl_mat =  9
    # Ar_mat = 10
    # K_mat  = 11
    # Ca_mat = 12
    # --> e.g. for water comp_vector = [11.11, 0, 0, 88.89, 0, 0, 0, 0, 0, 0, 0, 0]
    #- model is number in {0,1,2} to specify model for cross section computation 
    # model = 0 -> 2.15 from Uilkema
    # model = 1 -> Moliere
    # model = 2 -> Geant
    #
    # The way this is calculated is:
    #
    #Note: Equation 2.15 from Uilkema is the microscopic elastic scatter
    #       cs. In the proton transport equation one needs to
    #       use the macroscopic cs. This is obtained as
    #       Macroscopic cs = atomic density x microscopic cs
    #Some physical constants:
    #m_p = 1.67262192e-27 #proton mass in kg
    #m_e =  9.1093837015e-31 #electron mass in kg
    g_to_MeV_per_c_sq = 1.0/(1.78266192e-36*1.0e9) #factor to transform units from grams to MeV/c^2?
    m_p = 938.2720813  #MeV/c^2
    m_e = 0.5109989461 #MeV/c^2
    N_A = 6.02214076e23 #avogadro constant
    alpha = 0.0072973525 #fine-structure constant (unitless)
    ee = 1.602176634e-19 #elementary electric charge
    eps0 = 1.418284572502546e-26 #elektrische Feldkonstante ([A s/ Vm ]=[F/m])
    h_bar_x_c = hbc = 0.19732697e-10 #MeV·m
    #atomic numbers
    Z_array = [1, 6, 7, 8, 11, 12, 15, 16, 17, 18, 19, 20]
    #atomic weights
    Mol_weights = [1.008, 12.011, 14.007, 15.999, 22.989, 24.305, 30.973, 32.060, 35.450, 39.948, 39.098, 40.078] #g/Mol;
    #Ionization energies
    #I_eV  = vcat(19.0, (11.2 .+ 11.7.*Z_array[2:6]), (52.8 .+ 8.71.*Z_array[7:12]))
    #I_MeV = 1.0e-6 .* I_eV
    #ln_I_MeV = log.(I_MeV)

    #find number of unique material to reduce comp. effort
    idx = Base.unique(i -> rho[i], 1:length(rho))
    Sigma_e = zeros(size(E_MeV,1),size(mu_0,1),size(idx,1))
    for k=1:length(idx)
        for j=1:size(mu_0,1)
            N_i = rho[idx[k]] .* (comp_vector[:,idx[k]] ./ 100) .* N_A ./ Mol_weights
            m_t_i = (Mol_weights ./ N_A) .* g_to_MeV_per_c_sq
            m_t_i = m_t_i - Z_array * m_e #this is the mass of only the nucleus of the target atom in MeV/c²
            com_to_lab = (1.0 .+ mu_0[j]*2.0*m_p./m_t_i .+ (m_p./m_t_i).^2).^(3.0/2.0) ./ (1.0 .+ mu_0[j].*m_p./m_t_i)

            if model == 0 
                for i=1:size(E_MeV,1)
                    if E_MeV[i] == 0
                        Sigma_e[i,j,k] = 0
                    else
                        v_p = sqrt(2.0 * E_MeV[i] / m_p)

                        m_0_i = m_p .* m_t_i ./ (m_p .+ m_t_i)
                        eta_i = (Z_array.^(1.0/3.0) .* alpha .* m_e ./ (m_p * v_p)).^2
                    
                        F1_i = (Z_array .* ee^2 ./ (4.0 * pi * eps0 * m_0_i * v_p^2)).^2
                        F2_i = 1.0 ./ (1.0 .- mu_0[j] .+ 2.0 .* eta_i).^2
                    
                        Sigma_e_i = (F1_i .* F2_i .* com_to_lab) .* N_i
                        Sigma_e[i,j,k] =  sum(Sigma_e_i)
                    end
                end
            elseif model == 1
                for i=1:size(E_MeV,1)
                    if E_MeV[i] == 0
                        Sigma_e[i,j,k] = 0
                    else
                        gamma = E_MeV[i]  / m_p + 1.0
                        p_x_c = m_p * sqrt(gamma^2 - 1.0)

                        #alpha squared = (z Z / 137 * beta)^2 with z = 1 for protons
                        alpha_sq = (Z_array .* alpha).^2 ./ (1.0 .- 1.0 ./ gamma.^2)

                        #(1 / k)^2 = (h_bar / p)^2 = (h_bar c / p c)^2
                        inv_k_sq = (h_bar_x_c / p_x_c)^2

                        #Shielding factor
                        chi_0_sq = (1.13 .* alpha .* Z_array.^(1.0/3.0) .* m_e ./ p_x_c).^2
                        chi_alpha_sq = chi_0_sq .* (1.13 .+ 3.76 .* alpha_sq)

                        #Factor that accounts for electronic screening
                        q_screen = 1.0 ./ (1.0 .- mu_0[j] .+ chi_alpha_sq).^2

                        Sigma_e_i = 4.0 .* N_i .* alpha_sq .* inv_k_sq .* q_screen
                        Sigma_e_i = Sigma_e_i .* com_to_lab
                        Sigma_e[i,j,k] =  sum(Sigma_e_i)
                    end
                end
            elseif model == 2
                for i=1:size(E_MeV,1)
                    if E_MeV[i] == 0
                        Sigma_e[i,j,k] = 0
                    else
                        #gamma = E/(m_p c^2). m_p in units of MeV/c^2 -> gamma dimensionless
                        gamma = E_MeV[i] / m_p + 1.0

                        #momentum times speed of light
                        p_x_c = m_p * sqrt(gamma^2 - 1.0)

                        #alpha squared = (z Z / 137 * beta)^2 with z = 1 for protons
                        alpha_sq = (Z_array .* alpha).^2 ./ (1.0 .- 1.0./gamma.^2)

                        #(1 / k)^2 = (h_bar / p)^2 = (h_bar c / p c)^2
                        inv_k_sq = (h_bar_x_c/p_x_c)^2

                        #Shielding factor
                        chi_0_sq = (1.13 .* alpha .* Z_array.^(1.0/3.0) .* m_e ./ p_x_c).^2
                        chi_alpha_sq = chi_0_sq .* (1.13 .+ 3.76 .* alpha_sq)

                        #Factor that accounts for electronic screening
                        q_screen = 1.0./(1.0 .- mu_0[j] .+ chi_alpha_sq).^2

                        #Factor for nuclear size
                        R_N = 1.27 .* Mol_weights.^(0.27)
                        q_N = 1.0 ./ R_N
                        a_A = 3.0 .* q_N.^2 ./ E_MeV[i]^2

                        q_p = 232.0 # MeV
                        a_p = 3.0 * q_p^2/ E_MeV[i]^2
                        F_N = (1.0 .- 1.0 ./ Z_array) .* exp.(-(1.0 .- mu_0[j]).^2 ./ a_A.^2) .+ 1.0 ./ Z_array .* exp(-(1.0 .- mu_0[j]).^2 ./ a_p.^2)

                        Sigma_e_i = 4.0 * N_i .* alpha_sq .* inv_k_sq .* q_screen .* F_N
                        Sigma_e_i = Sigma_e_i .* com_to_lab
                        Sigma_e[i,j,k] = sum(Sigma_e_i)
                    end
                end
            else
                println("No valid model number entered!")
            end 
        end
    end
    return Sigma_e, idx
end

struct MaterialParametersProtons
    # tabulated values for stopping power in water and associated energies
    S_tab::Array{Float64,2};
    E_tab::Array{Float64,1};

    sigma_ce::Array{Float64,3};
    sigma_xi::Array{Float64,3};
    comp_vector::Array{Float64,2};

    # constructor
    function MaterialParametersProtons(settings::Settings,OmegaMin=5)
        E_tab_PSTAR = [0.00100000000000000	0.00125000000000000	0.00150000000000000	0.00175000000000000	0.00200000000000000	0.00250000000000000	0.00300000000000000	0.00350000000000000	0.00400000000000000	0.00450000000000000	0.00500000000000000	0.00600000000000000	0.00700000000000000	0.00800000000000000	0.00900000000000000	0.0100000000000000	0.0125000000000000	0.0150000000000000	0.0175000000000000	0.0200000000000000	0.0250000000000000	0.0300000000000000	0.0350000000000000	0.0400000000000000	0.0450000000000000	0.0500000000000000	0.0600000000000000	0.0700000000000000	0.0800000000000000	0.0900000000000000	0.100000000000000	0.125000000000000	0.150000000000000	0.175000000000000	0.200000000000000	0.250000000000000	0.300000000000000	0.350000000000000	0.400000000000000	0.450000000000000	0.500000000000000	0.600000000000000	0.700000000000000	0.800000000000000	0.900000000000000	1	1.25000000000000	1.50000000000000	1.75000000000000	2	2.50000000000000	3	3.50000000000000	4	4.50000000000000	5	6	7	8	9	10	12.5000000000000	15	17.5000000000000	20	25	30	35	40	45	50	60	70	80	90	100	125	150	175	200	250	300	350	400	450	500	600	700	800	900]';
        S_tab_PSTAR = [176.900000000000	187.800000000000	198.400000000000	208.600000000000	218.400000000000	237	254.400000000000	270.800000000000	286.400000000000	301.200000000000	315.300000000000	342	366.700000000000	390	412	432.900000000000	474.500000000000	511	543.700000000000	573.300000000000	624.500000000000	667.100000000000	702.800000000000	732.400000000000	756.900000000000	776.800000000000	805	820.500000000000	826	823.900000000000	816.100000000000	781.400000000000	737.100000000000	696.900000000000	661.300000000000	600.600000000000	550.400000000000	508	471.900000000000	440.600000000000	413.200000000000	368	332.500000000000	303.900000000000	280.500000000000	260.800000000000	222.900000000000	195.700000000000	174.900000000000	158.600000000000	134.400000000000	117.200000000000	104.200000000000	94.0400000000000	85.8600000000000	79.1100000000000	68.5800000000000	60.7100000000000	54.6000000000000	49.6900000000000	45.6700000000000	38.1500000000000	32.9200000000000	29.0500000000000	26.0700000000000	21.7500000000000	18.7600000000000	16.5600000000000	14.8800000000000	13.5400000000000	12.4500000000000	10.7800000000000	9.55900000000000	8.62500000000000	7.88800000000000	7.28900000000000	6.19200000000000	5.44500000000000	4.90300000000000	4.49200000000000	3.91100000000000	3.52000000000000	3.24100000000000	3.03200000000000	2.87100000000000	2.74300000000000	2.55600000000000	2.42600000000000	2.33300000000000	2.26400000000000];
        
        E_rest = 938.26 #MeV proton rest energy

        if settings.waterEq 
            rho = settings.density[:]
            idx = Base.unique(i -> rho[i], 1:length(rho))
            rho = rho[idx]
            comp_vector = [11.11 0 0 88.89 0 0 0 0 0 0 0 0]'#H2O
            sigma_tmp = ExtendedTransportCorrection(settings.nPN,E_tab_PSTAR,1,comp_vector,1,OmegaMin); sigma_ce = zeros(size(sigma_tmp,1),size(sigma_tmp,2),1); sigma_ce[:,:,1] = sigma_tmp;
            sigma_tmp = TransportCoefficientsGFP2(settings.nPN,E_tab_PSTAR,1,comp_vector,1,OmegaMin); sigma_xi  = zeros(size(sigma_tmp,1),size(sigma_tmp,3),1); sigma_xi[:,:,1] = sigma_tmp; 
            sigma_tmp = nothing

            S_tab_PSTAR = get_S(E_tab_PSTAR,rho,comp_vector,"testCase_CSD/data/proton_S_data_topas")
            E_tab_PSTAR= dropdims(E_tab_PSTAR, dims = tuple(findall(size(E_tab_PSTAR) .== 1)...)).+ E_rest
        else
            rho = settings.density[:]
            idx = Base.unique(i -> rho[i], 1:length(rho))
            rho = rho[idx]
            comp_vector = matComp(settings.densityHU[idx])
            sigma_ce = ExtendedTransportCorrection(settings.nPN,E_tab_PSTAR,rho,comp_vector,1,OmegaMin,true)
            sigma_xi = TransportCoefficientsGFP2(settings.nPN,E_tab_PSTAR,rho,comp_vector,1,OmegaMin,true) 
            S_tab_PSTAR = get_S(E_tab_PSTAR,rho,comp_vector,"testCase_CSD/data/proton_S_data_topas") # p_S_Bethe(E_tab_PSTAR,rho,comp_vector)
            E_tab_PSTAR= dropdims(E_tab_PSTAR, dims = tuple(findall(size(E_tab_PSTAR) .== 1)...)).+ E_rest
        end
        println("Computed cross sections and stopping powers")
        new(S_tab_PSTAR,E_tab_PSTAR,sigma_ce, sigma_xi,comp_vector);
     end

    function TransportCoefficientsGFP2(N,E,rho,comp_vector,model,OmegaMin,perMat=false) 
        mu, w = gausslegendre(1000);
        theta = acosd.(mu)
        nMu = length(mu);
        nE = length(E);
        D=2
        if perMat
            sigma_c,idx = Sigma_eModels_perMat(E,mu,model)
            rho = ones(12)
            comp_vector = Diagonal(ones(12)*100)
        else
            sigma_c,idx = Sigma_eModels(E,mu,rho,comp_vector,model)
        end

        nIdx = size(comp_vector,2)
        xi = zeros(nE,nIdx,D)

        for k=1:nIdx
            for n = 1:nE
                for d=1:D
                    xi[n,k,d] = 2*pi*dot(w, sigma_c[n,:,k].* (1 .- mu).^d)
                end
            end
        end
        
        return xi
    end

    function ExtendedTransportCorrection(N,E,rho,comp_vector,model,OmegaMin,perMat=false) 
        mu, w = gausslegendre(1000);
        nMu = length(mu);
        nE = length(E);
    
        if perMat
            sigma_c,idx = Sigma_eModels_perMat(E,mu,model)
            comp_vector = Diagonal(ones(12)*100)
        else
           sigma_c,_ = Sigma_eModels(E,mu,rho,comp_vector,model)
        end
        
        nIdx = size(comp_vector,2)
        xi_e = zeros(nE,N+2,nIdx)

        # Precompute Legendre Polynomials
        Pl_mu = reduce(hcat,collectPl.(mu, lmax = N+1))

        for k=1:nIdx
            for n = 1:nE
                for i = 1:N+2
                        xi_e[n,i,k] = 2*pi*dot(w,sigma_c[n,:,k].*Pl_mu[i,:])
                end
                xi_e[n,:,k] .-= xi_e[n ,N+2,k]'
            end
        end
         return xi_e[:,1:N+1,:] 
    end

    function HUtoDensity(HU::Array{T,1},entryType::String="HU") where {T<:AbstractFloat}
        #This routine is based on that of the raytracer, which uses paper:
        # Schneider, Bortfeld and Schlegel
        # Correlation between CT numbers and tissue parameters needed for Monte Carlo
        # simulations of clinical dose distributions.
        # Phys. Med. Biol. 45 (2000), pp. 459-478.
        rho_air = 1.21E-3
        rho_adipose = 0.93

        rho_values = zeros(length(HU))
        for  i=1:length(HU)
            if     ( (HU[i] >= -1000)&&(HU[i] <=   -98) ) 
                rho_values[i] = ((HU[i] + 98.0)/(98.0 - 1000.0)) * rho_air + ((HU[i] + 1000.0)/(1000.0 - 98.0)) * rho_adipose
            elseif ( (HU[i] >    -98)&&(HU[i] <=   +14) ) 
                rho_values[i] = 1.0180 + 0.893e-3 * HU[i]
            elseif ( (HU[i] >    +14)&&(HU[i] <=   +23) ) 
                rho_values[i] = 1.030
            elseif ( (HU[i] >    +23)&&(HU[i] <=  +100) ) 
                rho_values[i] = 1.0030 + 1.169e-3 * HU[i]
            elseif ( (HU[i] >   +100)&&(HU[i] <= +1600) ) 
                rho_values[i] = 1.0170 + 0.592e-3 * HU[i]
            else
            println("CT_value out of range")
            end
        end
        return rho_values
    end


    function matComp(HU::Array{T,1},entryType::String="HU") where {T<:AbstractFloat}
        #this is equivalent to the mat_comp.f90 class in tracer (adapted to requirements of DLRA code)
        #computes density and material composition for given HU CT values
        HU_min = -1000
        HU_max = +1600

        #material info
        no_comp_nuclides = 12
        H_mat  =  1
        C_mat  =  2
        N_mat  =  3
        O_mat  =  4
        Na_mat =  5
        Mg_mat =  6
        P_mat  =  7
        S_mat  =  8
        Cl_mat =  9
        Ar_mat = 10
        K_mat  = 11
        Ca_mat = 12

        rho_H  = 0.00008988  #8.3748E-05
        rho_C  = 2.267   #2.0
        rho_N  = 0.00125   #0.0011652
        rho_O  = 0.00143   #0.00133151
        rho_Na = 0.97   #0.971
        rho_Mg = 1.74   #1.74
        rho_P  = 1.82   #2.2
        rho_S  = 2.067   #2.07
        rho_Cl = 0.003   #0.00299473
        rho_Ar = 0.0017837   #0.00166201
        rho_K  = 0.89   #0.862
        rho_Ca = 1.54   #1.55

        #Atomic numbers
        Z_array = [1, 6, 7, 8, 11, 12, 15, 16, 17, 18, 19, 20]
        #Atomic weights
        Mol_weights = [1.008, 12.011, 14.007, 15.999, 22.989, 24.305, 30.973, 32.060, 35.450, 39.948, 39.098, 40.078]
        #Ionization energies
        I_eV  = vcat(19.0, (11.2 .+ 11.7.*Z_array[2:6]), (52.8 .+ 8.71.*Z_array[7:12]))
        I_MeV = 1.0E-6 .* I_eV
        #Ln of ionization energies
        ln_I_MeV = log.(I_MeV)

        rho_air = 1.21E-3
        rho_adipose = 0.93

        #compute comp. vector
        comp_vector = zeros(no_comp_nuclides,length(HU))

        for  i=1:length(HU)
            if     ( (HU[i] >= -1000)&&(HU[i] <=  -950) ) 
                comp_vector[H_mat ,i] =  0.0
                comp_vector[C_mat ,i] =  0.0
                comp_vector[N_mat ,i] = 75.5
                comp_vector[O_mat ,i] = 23.3
                comp_vector[Na_mat ,i] =  0.0
                comp_vector[Mg_mat ,i] =  0.0
                comp_vector[P_mat ,i] =  0.0
                comp_vector[S_mat ,i] =  0.0
                comp_vector[Cl_mat ,i] =  0.0
                comp_vector[Ar_mat ,i] =  1.3
                comp_vector[K_mat ,i] =  0.0
                comp_vector[Ca_mat ,i] =  0.0
            elseif ( (HU[i] >   -950)&&(HU[i] <=  -120) ) 
                comp_vector[H_mat ,i] = 10.3
                comp_vector[C_mat ,i] = 10.5
                comp_vector[N_mat ,i] =  3.1
                comp_vector[O_mat ,i] = 74.9
                comp_vector[Na_mat ,i] =  0.2
                comp_vector[Mg_mat ,i] =  0.0
                comp_vector[P_mat ,i] =  0.2
                comp_vector[S_mat ,i] =  0.3
                comp_vector[Cl_mat ,i] =  0.3
                comp_vector[Ar_mat ,i] =  0.0
                comp_vector[K_mat ,i] =  0.2
                comp_vector[Ca_mat ,i] =  0.0
            elseif ( (HU[i] >   -120)&&(HU[i] <=   -83) ) 
                comp_vector[H_mat ,i] = 11.6
                comp_vector[C_mat ,i] = 68.1
                comp_vector[N_mat ,i] =  0.2
                comp_vector[O_mat ,i] = 19.8
                comp_vector[Na_mat ,i] =  0.1
                comp_vector[Mg_mat ,i] =  0.0
                comp_vector[P_mat ,i] =  0.0
                comp_vector[S_mat ,i] =  0.1
                comp_vector[Cl_mat ,i] =  0.1
                comp_vector[Ar_mat ,i] =  0.0
                comp_vector[K_mat ,i] =  0.0
                comp_vector[Ca_mat ,i] =  0.0
            elseif ( (HU[i] >    -83)&&(HU[i] <=   -53) ) 
                comp_vector[H_mat ,i] = 11.3
                comp_vector[C_mat ,i] = 56.7
                comp_vector[N_mat ,i] =  0.9
                comp_vector[O_mat ,i] = 30.8
                comp_vector[Na_mat ,i] =  0.1
                comp_vector[Mg_mat ,i] =  0.0
                comp_vector[P_mat ,i] =  0.0
                comp_vector[S_mat ,i] =  0.1
                comp_vector[Cl_mat ,i] =  0.1
                comp_vector[Ar_mat ,i] =  0.0
                comp_vector[K_mat ,i] =  0.0
                comp_vector[Ca_mat ,i] =  0.0
            elseif ( (HU[i] >    -53)&&(HU[i] <=   -23) ) 
                comp_vector[H_mat ,i] = 11.0
                comp_vector[C_mat ,i] = 45.8
                comp_vector[N_mat ,i] =  1.5
                comp_vector[O_mat ,i] = 41.1
                comp_vector[Na_mat ,i] =  0.1
                comp_vector[Mg_mat ,i] =  0.0
                comp_vector[P_mat ,i] =  0.1
                comp_vector[S_mat ,i] =  0.2
                comp_vector[Cl_mat ,i] =  0.2
                comp_vector[Ar_mat ,i] =  0.0
                comp_vector[K_mat ,i] =  0.0
                comp_vector[Ca_mat ,i] =  0.0
            elseif ( (HU[i] >    -23)&&(HU[i] <=    +7) ) 
                comp_vector[H_mat ,i] = 10.8
                comp_vector[C_mat ,i] = 35.6
                comp_vector[N_mat ,i] =  2.2
                comp_vector[O_mat ,i] = 50.9
                comp_vector[Na_mat ,i] =  0.0
                comp_vector[Mg_mat ,i] =  0.0
                comp_vector[P_mat ,i] =  0.1
                comp_vector[S_mat ,i] =  0.2
                comp_vector[Cl_mat ,i] =  0.2
                comp_vector[Ar_mat ,i] =  0.0
                comp_vector[K_mat ,i] =  0.0
                comp_vector[Ca_mat ,i] =  0.0
            elseif ( (HU[i] >     +7)&&(HU[i] <=   +18) ) 
                comp_vector[H_mat ,i] = 10.6
                comp_vector[C_mat ,i] = 28.4
                comp_vector[N_mat ,i] =  2.6
                comp_vector[O_mat ,i] = 57.8
                comp_vector[Na_mat ,i] =  0.0
                comp_vector[Mg_mat ,i] =  0.0
                comp_vector[P_mat ,i] =  0.1
                comp_vector[S_mat ,i] =  0.2
                comp_vector[Cl_mat ,i] =  0.2
                comp_vector[Ar_mat ,i] =  0.0
                comp_vector[K_mat ,i] =  0.1
                comp_vector[Ca_mat ,i] =  0.0
            elseif ( (HU[i] >    +18)&&(HU[i] <=   +80) ) 
                comp_vector[H_mat ,i] = 10.3
                comp_vector[C_mat ,i] = 13.4
                comp_vector[N_mat ,i] =  3.0
                comp_vector[O_mat ,i] = 72.3
                comp_vector[Na_mat ,i] =  0.2
                comp_vector[Mg_mat ,i] =  0.0
                comp_vector[P_mat ,i] =  0.2
                comp_vector[S_mat ,i] =  0.2
                comp_vector[Cl_mat ,i] =  0.2
                comp_vector[Ar_mat ,i] =  0.0
                comp_vector[K_mat ,i] =  0.2
                comp_vector[Ca_mat ,i] =  0.0
            elseif ( (HU[i] >    +80)&&(HU[i]<=  +120) ) 
                comp_vector[H_mat ,i] =  9.4
                comp_vector[C_mat ,i] = 20.7
                comp_vector[N_mat ,i] =  6.2
                comp_vector[O_mat ,i] = 62.2
                comp_vector[Na_mat ,i] =  0.6
                comp_vector[Mg_mat ,i] =  0.0
                comp_vector[P_mat ,i] =  0.0
                comp_vector[S_mat ,i] =  0.6
                comp_vector[Cl_mat ,i] =  0.3
                comp_vector[Ar_mat ,i] =  0.0
                comp_vector[K_mat ,i] =  0.0
                comp_vector[Ca_mat ,i] =  0.0
            elseif ( (HU[i] >   +120)&&(HU[i] <=  +200) ) 
                comp_vector[H_mat ,i] =  9.5
                comp_vector[C_mat ,i] = 45.5
                comp_vector[N_mat ,i] =  2.5
                comp_vector[O_mat ,i] = 35.5
                comp_vector[Na_mat ,i] =  0.1
                comp_vector[Mg_mat ,i] =  0.0
                comp_vector[P_mat ,i] =  2.1
                comp_vector[S_mat ,i] =  0.1
                comp_vector[Cl_mat ,i] =  0.1
                comp_vector[Ar_mat ,i] =  0.0
                comp_vector[K_mat ,i] =  0.1
                comp_vector[Ca_mat ,i] =  4.5
            elseif ( (HU[i]>   +200)&&(HU[i] <=  +300) ) 
                comp_vector[H_mat ,i] =  8.9
                comp_vector[C_mat ,i] = 42.3
                comp_vector[N_mat ,i] =  2.7
                comp_vector[O_mat ,i] = 36.3
                comp_vector[Na_mat ,i] =  0.1
                comp_vector[Mg_mat ,i] =  0.0
                comp_vector[P_mat ,i] =  3.0
                comp_vector[S_mat ,i] =  0.1
                comp_vector[Cl_mat ,i] =  0.1
                comp_vector[Ar_mat ,i] =  0.0
                comp_vector[K_mat ,i] =  0.1
                comp_vector[Ca_mat ,i] =  6.4
            elseif ( (HU[i] >   +300)&&(HU[i] <=  +400) ) 
                comp_vector[H_mat ,i] =  8.2
                comp_vector[C_mat ,i] = 39.1
                comp_vector[N_mat ,i] =  2.9
                comp_vector[O_mat ,i] = 37.2
                comp_vector[Na_mat ,i] =  0.1
                comp_vector[Mg_mat ,i] =  0.0
                comp_vector[P_mat ,i] =  3.9
                comp_vector[S_mat ,i] =  0.1
                comp_vector[Cl_mat ,i] =  0.1
                comp_vector[Ar_mat ,i] =  0.0
                comp_vector[K_mat ,i] =  0.1
                comp_vector[Ca_mat ,i] =  8.3
            elseif ( (HU[i] >   +400)&&(HU[i] <=  +500) ) 
                comp_vector[H_mat ,i] =  7.6
                comp_vector[C_mat ,i] = 36.1
                comp_vector[N_mat ,i] =  3.0
                comp_vector[O_mat ,i] = 38.0
                comp_vector[Na_mat ,i] =  0.1
                comp_vector[Mg_mat ,i] =  0.1
                comp_vector[P_mat ,i] =  4.7
                comp_vector[S_mat ,i] =  0.2
                comp_vector[Cl_mat ,i] =  0.1
                comp_vector[Ar_mat ,i] =  0.0
                comp_vector[K_mat ,i] =  0.0
                comp_vector[Ca_mat ,i] = 10.1
            elseif ( (HU[i] >   +500)&&(HU[i] <=  +600) ) 
                comp_vector[H_mat ,i] =  7.1
                comp_vector[C_mat ,i] = 33.5
                comp_vector[N_mat ,i] =  3.2
                comp_vector[O_mat ,i] = 38.7
                comp_vector[Na_mat ,i] =  0.1
                comp_vector[Mg_mat ,i] =  0.1
                comp_vector[P_mat ,i] =  5.4
                comp_vector[S_mat ,i] =  0.2
                comp_vector[Cl_mat ,i] =  0.0
                comp_vector[Ar_mat ,i] =  0.0
                comp_vector[K_mat ,i] =  0.0
                comp_vector[Ca_mat ,i] = 11.7
            elseif ( (HU[i]>   +600)&&(HU[i] <=  +700) ) 
                comp_vector[H_mat ,i] =  6.6
                comp_vector[C_mat ,i] = 31.0
                comp_vector[N_mat ,i] =  3.3
                comp_vector[O_mat ,i] = 39.4
                comp_vector[Na_mat ,i] =  0.1
                comp_vector[Mg_mat ,i] =  0.1
                comp_vector[P_mat ,i] =  6.1
                comp_vector[S_mat ,i] =  0.2
                comp_vector[Cl_mat ,i] =  0.0
                comp_vector[Ar_mat ,i] =  0.0
                comp_vector[K_mat ,i] =  0.0
                comp_vector[Ca_mat ,i] = 13.2
            elseif ( (HU[i] >   +700)&&(HU[i] <=  +800) ) 
                comp_vector[H_mat ,i] =  6.1
                comp_vector[C_mat ,i] = 28.7
                comp_vector[N_mat ,i] =  3.5
                comp_vector[O_mat ,i] = 40.0
                comp_vector[Na_mat ,i] =  0.1
                comp_vector[Mg_mat ,i] =  0.1
                comp_vector[P_mat ,i] =  6.7
                comp_vector[S_mat ,i] =  0.2
                comp_vector[Cl_mat ,i] =  0.0
                comp_vector[Ar_mat ,i] =  0.0
                comp_vector[K_mat ,i] =  0.0
                comp_vector[Ca_mat ,i] = 14.6
            elseif ( (HU[i]>   +800)&&(HU[i]<=  +900) ) 
                comp_vector[H_mat ,i] =  5.6
                comp_vector[C_mat ,i] = 26.5
                comp_vector[N_mat ,i] =  3.6
                comp_vector[O_mat ,i] = 40.5
                comp_vector[Na_mat ,i] =  0.1
                comp_vector[Mg_mat ,i] =  0.2
                comp_vector[P_mat ,i] =  7.3
                comp_vector[S_mat ,i] =  0.3
                comp_vector[Cl_mat ,i] =  0.0
                comp_vector[Ar_mat ,i] =  0.0
                comp_vector[K_mat ,i] =  0.0
                comp_vector[Ca_mat ,i] = 15.9
            elseif ( (HU[i] >   +900)&&(HU[i] <= +1000) ) 
                comp_vector[H_mat ,i] =  5.2
                comp_vector[C_mat ,i] = 24.6
                comp_vector[N_mat ,i] =  3.7
                comp_vector[O_mat ,i] = 41.1
                comp_vector[Na_mat ,i] =  0.1
                comp_vector[Mg_mat ,i] =  0.2
                comp_vector[P_mat ,i] =  7.8
                comp_vector[S_mat ,i] =  0.3
                comp_vector[Cl_mat ,i] =  0.0
                comp_vector[Ar_mat ,i] =  0.0
                comp_vector[K_mat ,i] =  0.0
                comp_vector[Ca_mat ,i] = 17.0
            elseif ( (HU[i]>  +1000)&&(HU[i]<= +1100) ) 
                comp_vector[H_mat ,i] =  4.9
                comp_vector[C_mat ,i] = 22.7
                comp_vector[N_mat ,i] =  3.8
                comp_vector[O_mat ,i] = 41.6
                comp_vector[Na_mat ,i] =  0.1
                comp_vector[Mg_mat ,i] =  0.2
                comp_vector[P_mat ,i] =  8.3
                comp_vector[S_mat ,i] =  0.3
                comp_vector[Cl_mat ,i] =  0.0
                comp_vector[Ar_mat ,i] =  0.0
                comp_vector[K_mat ,i] =  0.0
                comp_vector[Ca_mat ,i] = 18.1
            elseif ( (HU[i]>  +1100)&&(HU[i]<= +1200) ) 
                comp_vector[H_mat ,i] =  4.5
                comp_vector[C_mat ,i] = 21.0
                comp_vector[N_mat ,i] =  3.9
                comp_vector[O_mat ,i] = 42.0
                comp_vector[Na_mat ,i] =  0.1
                comp_vector[Mg_mat ,i] =  0.2
                comp_vector[P_mat ,i] =  8.8
                comp_vector[S_mat ,i] =  0.3
                comp_vector[Cl_mat ,i] =  0.0
                comp_vector[Ar_mat ,i] =  0.0
                comp_vector[K_mat ,i] =  0.0
                comp_vector[Ca_mat ,i] = 19.2
            elseif ( (HU[i]>  +1200)&&(HU[i]<= +1300) ) 
                comp_vector[H_mat ,i] =  4.2
                comp_vector[C_mat ,i] = 19.4
                comp_vector[N_mat ,i] =  4.0
                comp_vector[O_mat ,i] = 42.5
                comp_vector[Na_mat ,i] =  0.1
                comp_vector[Mg_mat ,i] =  0.2
                comp_vector[P_mat ,i] =  9.2
                comp_vector[S_mat ,i] =  0.3
                comp_vector[Cl_mat ,i] =  0.0
                comp_vector[Ar_mat ,i] =  0.0
                comp_vector[K_mat ,i] =  0.0
                comp_vector[Ca_mat ,i] = 20.1
            elseif ( (HU[i] >  +1300)&&(HU[i] <= +1400) ) 
                comp_vector[H_mat ,i] =  3.9
                comp_vector[C_mat ,i] = 17.9
                comp_vector[N_mat ,i] =  4.1
                comp_vector[O_mat ,i] = 42.9
                comp_vector[Na_mat ,i] =  0.1
                comp_vector[Mg_mat ,i] =  0.2
                comp_vector[P_mat ,i] =  9.6
                comp_vector[S_mat ,i] =  0.3
                comp_vector[Cl_mat ,i] =  0.0
                comp_vector[Ar_mat ,i] =  0.0
                comp_vector[K_mat ,i] =  0.0
                comp_vector[Ca_mat ,i] = 21.0
            elseif ( (HU[i]>  +1400)&&(HU[i] <= +1500) ) 
                comp_vector[H_mat ,i] =  3.6
                comp_vector[C_mat ,i] = 16.5
                comp_vector[N_mat ,i] =  4.2
                comp_vector[O_mat ,i] = 43.2
                comp_vector[Na_mat ,i] =  0.1
                comp_vector[Mg_mat ,i] =  0.2
                comp_vector[P_mat ,i] = 10.0
                comp_vector[S_mat ,i] =  0.3
                comp_vector[Cl_mat ,i] =  0.0
                comp_vector[Ar_mat ,i] =  0.0
                comp_vector[K_mat ,i] =  0.0
                comp_vector[Ca_mat ,i] = 21.9
            elseif ( (HU[i] >  +1500)&&(HU[i] <= +1600) ) 
                comp_vector[H_mat ,i] =  3.4
                comp_vector[C_mat ,i] = 15.5
                comp_vector[N_mat ,i] =  4.2
                comp_vector[O_mat ,i] = 43.5
                comp_vector[Na_mat ,i] =  0.1
                comp_vector[Mg_mat ,i] =  0.2
                comp_vector[P_mat ,i] = 10.3
                comp_vector[S_mat ,i] =  0.3
                comp_vector[Cl_mat ,i] =  0.0
                comp_vector[Ar_mat ,i] =  0.0
                comp_vector[K_mat ,i] =  0.0
                comp_vector[Ca_mat ,i] = 22.5
            else
                println("CT_value out of range")
            end
        end
        return comp_vector
    end

    function get_S(E_MeV,rho,comp_vector,file_name="testCase_CSD/data/proton_S_data_PSTAR")
        #Gives stopping power based on data from PSTAR for 12 materials
        #
        #- E is energy in MeV
        #- rho is density
        #- comp_vector is vector with percentage wise composition (according to mass?) of following materials:
        # H_mat  =  1
        # C_mat  =  2
        # N_mat  =  3
        # O_mat  =  4
        # Na_mat =  5
        # Mg_mat =  6
        # P_mat  =  7
        # S_mat  =  8
        # Cl_mat =  9
        # Ar_mat = 10
        # K_mat  = 11
        # Ca_mat = 12
        num_energies, energies, materials = read_stopping_power_file(file_name)
        S_vect = zeros(12,size(E_MeV,1))
        E2S = LinearInterpolation(energies, materials["H"][1]; extrapolation_bc=Flat())
        S_vect[1,:] = E2S.(E_MeV)
        E2S = LinearInterpolation(energies, materials["C"][1]; extrapolation_bc=Flat())
        S_vect[2,:] = E2S.(E_MeV)
        E2S = LinearInterpolation(energies, materials["N"][1]; extrapolation_bc=Flat())
        S_vect[3,:] = E2S.(E_MeV)
        E2S = LinearInterpolation(energies, materials["O"][1]; extrapolation_bc=Flat())
        S_vect[4,:] = E2S.(E_MeV)
        E2S = LinearInterpolation(energies, materials["Na"][1]; extrapolation_bc=Flat())
        S_vect[5,:] = E2S.(E_MeV)
        E2S = LinearInterpolation(energies, materials["Mg"][1]; extrapolation_bc=Flat())
        S_vect[6,:] = E2S.(E_MeV)
        E2S = LinearInterpolation(energies, materials["P"][1]; extrapolation_bc=Flat())
        S_vect[7,:] = E2S.(E_MeV)
        E2S = LinearInterpolation(energies, materials["S"][1]; extrapolation_bc=Flat())
        S_vect[8,:] = E2S.(E_MeV)
        E2S = LinearInterpolation(energies, materials["Cl"][1]; extrapolation_bc=Flat())
        S_vect[9,:] = E2S.(E_MeV)
        E2S = LinearInterpolation(energies, materials["Ar"][1]; extrapolation_bc=Flat())
        S_vect[10,:] = E2S.(E_MeV)
        E2S = LinearInterpolation(energies, materials["K"][1]; extrapolation_bc=Flat())
        S_vect[11,:] = E2S.(E_MeV)
        E2S = LinearInterpolation(energies, materials["Ca"][1]; extrapolation_bc=Flat())
        S_vect[12,:] = E2S.(E_MeV)

        idx = Base.unique(i -> rho[i], 1:length(rho))
        stp = zeros(size(E_MeV,1),length(idx))
        for k=1:length(idx)
            stp[:,k] = (comp_vector[:,idx[k]]'*S_vect) * rho[idx[k]] / 100.0
        end
        return stp
    end 

    function read_stopping_power_file(filename)
        # Open the file and read its contents
        open(filename, "r") do io
            # Read the first line to get the number of energies
            num_energies = parse(Int, readline(io))
            
            # Read the energy values (one value per line)
            energies = Float64[]
            for _ in 1:num_energies
                push!(energies, parse(Float64, readline(io)))
            end
            
            # Initialize storage for material data
            materials = Dict{String, Vector{Vector{Float64}}}()
            
            # Read the 12 material blocks
            for _ in 1:12
                # Read the material name (chemical sign)
                material = readline(io)
                
                # Read the stopping powers for this material
                stopping_powers = Float64[]
                for _ in 1:num_energies
                    push!(stopping_powers, parse(Float64, readline(io)))
                end
                
                # Store the stopping powers in the dictionary
                if haskey(materials, material)
                    push!(materials[material], stopping_powers)
                else
                    materials[material] = [stopping_powers]
                end
            end
            
            return num_energies, energies, materials
        end
    end
end

function Sigma_eModels_perMat(E_MeV,mu_0,model)
    #Gives the (macroscopic) elastic scatter cross section individually for each material 
    #
    #- E is energy in MeV
    #- mu is cosine of the deflection angle
    #- rho is density
    #- comp_vector is vector with percentage wise composition (according to mass?) of following materials:
    # H_mat  =  1
    # C_mat  =  2
    # N_mat  =  3
    # O_mat  =  4
    # Na_mat =  5
    # Mg_mat =  6
    # P_mat  =  7
    # S_mat  =  8
    # Cl_mat =  9
    # Ar_mat = 10
    # K_mat  = 11
    # Ca_mat = 12
    # --> e.g. for water comp_vector = [11.11, 0, 0, 88.89, 0, 0, 0, 0, 0, 0, 0, 0]
    #- model is number in {0,1,2} to specify model for cross section computation 
    # model = 0 -> 2.15 from Uilkema
    # model = 1 -> Moliere
    # model = 2 -> Geant
    #
    # The way this is calculated is:
    #
    #Note: Equation 2.15 from Uilkema is the microscopic elastic scatter
    #       cs. In the proton transport equation one needs to
    #       use the macroscopic cs. This is obtained as
    #       Macroscopic cs = atomic density x microscopic cs
    #Some physical constants:
    #m_p = 1.67262192e-27 #proton mass in kg
    #m_e =  9.1093837015e-31 #electron mass in kg
    g_to_MeV_per_c_sq = 1.0/(1.78266192e-36*1.0e9) #factor to transform units from grams to MeV/c^2?
    m_p = 938.2720813  #MeV/c^2
    m_e = 0.5109989461 #MeV/c^2
    N_A = 6.02214076e23 #avogadro constant
    alpha = 0.0072973525 #fine-structure constant (unitless)
    ee = 1.602176634e-19 #elementary electric charge
    eps0 = 1.418284572502546e-26 #elektrische Feldkonstante ([A s/ Vm ]=[F/m])
    h_bar_x_c = hbc = 0.19732697e-10 #MeV·m
    # material names/symbols
    matNames = ["H","C","N","O","Na","Mg","P","S","Cl","Ar","K","Ca"]
    #atomic numbers
    Z_array = [1, 6, 7, 8, 11, 12, 15, 16, 17, 18, 19, 20]
    #atomic weights
    Mol_weights = [1.008, 12.011, 14.007, 15.999, 22.989, 24.305, 30.973, 32.060, 35.450, 39.948, 39.098, 40.078] #g/Mol;
    #rho = [8.3748E-05, 2.0, 0.0011652, 0.00133151, 0.971, 1.74, 2.2, 2.0, 0.00299473, 0.00166201, 0.862, 1.55]; 
    rho = ones(12)
    comp_vector = Diagonal(ones(12)*100)
    #Ionization energies
    #I_eV  = vcat(19.0, (11.2 .+ 11.7.*Z_array[2:6]), (52.8 .+ 8.71.*Z_array[7:12]))
    #I_MeV = 1.0e-6 .* I_eV
    #ln_I_MeV = log.(I_MeV)

    Sigma_e = zeros(size(E_MeV,1),size(mu_0,1),12)
    for k=1:12
        for j=1:size(mu_0,1)
            N_i = rho[k] .* (comp_vector[k,:] ./ 100) .* N_A ./ Mol_weights
            m_t_i = (Mol_weights ./ N_A) .* g_to_MeV_per_c_sq
            m_t_i = m_t_i - Z_array * m_e
            com_to_lab = (1.0 .+ mu_0[j]*2.0*m_p./m_t_i .+ (m_p./m_t_i).^2).^(3.0/2.0) ./ (1.0 .+ mu_0[j].*m_p./m_t_i)

            if model == 0 
                for i=1:size(E_MeV,1)
                    if E_MeV[i] == 0
                        Sigma_e[i,j,k] = 0
                    else
                        v_p = sqrt(2.0 * E_MeV[i] / m_p)

                        m_0_i = m_p .* m_t_i ./ (m_p .+ m_t_i)
                        eta_i = (Z_array.^(1.0/3.0) .* alpha .* m_e ./ (m_p * v_p)).^2
                    
                        F1_i = (Z_array .* ee^2 ./ (4.0 * pi * eps0 * m_0_i * v_p^2)).^2
                        F2_i = 1.0 ./ (1.0 .- mu_0[j] .+ 2.0 .* eta_i).^2
                    
                        Sigma_e_i = (F1_i .* F2_i .* com_to_lab) .* N_i
                        Sigma_e[i,j,k] =  sum(Sigma_e_i)
                    end
                end
            elseif model == 1
                for i=1:size(E_MeV,1)
                    if E_MeV[i] == 0
                        Sigma_e[i,j,k] = 0
                    else
                        gamma = E_MeV[i]  / m_p + 1.0 #Lorentz factor from relation to relativistic kinetic energy E_k = (gamma-1)mc², and c=1 bc of natural untis
                        p_x_c = m_p * sqrt(gamma^2 - 1.0) #relativistic momentum from E² = (pc)²+ (mc²)² --> (pc)²=E²​−(mc²)² = (γmc²)²−(mc²)²=m²c²(γ²−1), and c=1 bc of natural untis

                        #alpha squared = (z Z / 137 * beta)^2 with z = 1 for protons, beta = v/c , then equivalence is clear from above comments
                        alpha_sq = (Z_array .* alpha).^2 ./ (1.0 .- 1.0 ./ gamma.^2)

                        #(1 / k)^2 = (h_bar / p)^2 = (h_bar c / p c)^2
                        inv_k_sq = (h_bar_x_c / p_x_c)^2

                        #Shielding factor
                        chi_0_sq = (1.13 .* alpha .* Z_array.^(1.0/3.0) .* m_e ./ p_x_c).^2
                        chi_alpha_sq = chi_0_sq .* (1.13 .+ 3.76 .* alpha_sq)

                        #Factor that accounts for electronic screening
                        q_screen = 1.0 ./ (1.0 .- mu_0[j] .+ chi_alpha_sq).^2

                        Sigma_e_i = 4.0 .* N_i .* alpha_sq .* inv_k_sq .* q_screen
                        Sigma_e_i = Sigma_e_i .* com_to_lab
                        Sigma_e[i,j,k] =  sum(Sigma_e_i)
                    end
                end
            elseif model == 2
                for i=1:size(E_MeV,1)
                    if E_MeV[i] == 0
                        Sigma_e[i,j,k] = 0
                    else
                        #gamma = E/(m_p c^2). m_p in units of MeV/c^2 -> gamma dimensionless
                        gamma = E_MeV[i] / m_p + 1.0

                        #momentum times speed of light
                        p_x_c = m_p * sqrt(gamma^2 - 1.0)

                        #alpha squared = (z Z / 137 * beta)^2 with z = 1 for protons
                        alpha_sq = (Z_array .* alpha).^2 ./ (1.0 .- 1.0./gamma.^2)

                        #(1 / k)^2 = (h_bar / p)^2 = (h_bar c / p c)^2
                        inv_k_sq = (h_bar_x_c/p_x_c)^2

                        #Shielding factor
                        chi_0_sq = (1.13 .* alpha .* Z_array.^(1.0/3.0) .* m_e ./ p_x_c).^2
                        chi_alpha_sq = chi_0_sq .* (1.13 .+ 3.76 .* alpha_sq)

                        #Factor that accounts for electronic screening
                        q_screen = 1.0./(1.0 .- mu_0[j] .+ chi_alpha_sq).^2

                        #Factor for nuclear size
                        R_N = 1.27 .* Mol_weights.^(0.27)
                        q_N = 1.0 ./ R_N
                        a_A = 3.0 .* q_N.^2 ./ E_MeV[i]^2

                        q_p = 232.0 # MeV
                        a_p = 3.0 * q_p^2/ E_MeV[i]^2
                        F_N = (1.0 .- 1.0 ./ Z_array) .* exp.(-(1.0 .- mu_0[j]).^2 ./ a_A.^2) .+ 1.0 ./ Z_array .* exp(-(1.0 .- mu_0[j]).^2 ./ a_p.^2)

                        Sigma_e_i = 4.0 * N_i .* alpha_sq .* inv_k_sq .* q_screen .* F_N
                        Sigma_e_i = Sigma_e_i .* com_to_lab
                        Sigma_e[i,j,k] = sum(Sigma_e_i)
                    end
                end
            else
                println("No valid model number entered!")
            end 
        end
    end
    return Sigma_e, matNames
end
