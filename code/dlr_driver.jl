include("settings.jl")
include("Solver.jl")
using LinearAlgebra

function SolveNaiveUnconventional(obj::Solver)
    # Get rank
    r=obj.settings.r;
    Nt=obj.settings.tEnd/obj.settings.dt;
    t=0;

    # Set up initial condition
    u = SetupIC(obj::Solver);

    # Low-rank approx of init data:
    X,S,W = svd(u);
    
    # rank-r truncation:
    X = X[:,1:r];
    W = W[:,1:r];
    S = Diagonal(S);
    S = S[1:r, 1:r];
    K = zeros(size(X));

    for n = 1:Nt

        ################## K-step ##################
        K .= X*S;

        K .= K .+ obj.settings.dt*F(obj,K*W',n)*W;

        XNew,STmp = qr(K);
        XNew = XNew[:,1:r];

        MUp = XNew' * X;

        ################## L-step ##################
        L = W*S';

        L .= L .+ obj.settings.dt*(X'*F(obj,X*L',n))';
                
        WNew,STmp = qr(L);
        WNew = WNew[:,1:r];

        NUp = WNew' * W;
        W .= WNew;
        X .= XNew;

        ################## S-step ##################
        S .= MUp*S*(NUp')

        S .= S .+ obj.settings.dt.*X'*F(obj,X*S*W',n)*W;
        
        t = t+obj.settings.dt;
    end

    # return end time and solution
    return t, X,S,W;

end

function SolveUnconventional(obj::Solver)
    # Get rank
    r=obj.settings.r;
    Nt=obj.settings.tEnd/obj.settings.dt;
    t=0;

    # Set up initial condition
    u = SetupIC(obj::Solver);

    # Low-rank approx of init data:
    X,S,W = svd(u);
    
    # rank-r truncation:
    X = X[:,1:r];
    W = W[:,1:r];
    S = Diagonal(S);
    S = S[1:r, 1:r];
    K = zeros(size(X));
    dx = obj.settings.dx
    dt = obj.settings.dt
    

    for n = 1:Nt

        ################## K-step ##################
        K .= X*S;

        G = ScatteringMatrix(obj);
        rhs_K = zeros(obj.settings.NCells,r)
        rhs_L = zeros(obj.settings.NCells,r)
        rhs_S = zeros(obj.settings.NCells,r)

        WAW = W'*obj.A*W
        for j=2:obj.settings.NCells-1
            rhs_K[j,:] = -1/(2*dt)*(K[j+1,:]'-2*K[j,:]'+K[j-1,:]')+0.5 * (K[j+1,:]-K[j-1,:])'*WAW/dx;
        end
        WGW =  W'*G*W

        rhs_K = - rhs_K .- K*obj.sigmaT  - obj.settings.sigmaS*K*WGW; 

        K .= K .+ obj.settings.dt*rhs_K;

        XNew,STmp = qr(K);
        XNew = XNew[:,1:r];

        MUp = XNew' * X;

        ################## L-step ##################
        L = W*S';

        LA = L'*obj.A';
        X_hat = zeros(obj.settings.r,obj.settings.r)
        X_hat_2 = zeros(obj.settings.r,obj.settings.r)
        
        for i=1:obj.settings.r
            for l=1:obj.settings.r
                for j=2:(obj.settings.NCells-1)
                     X_hat[i,l] += X[j,i]*(X[j+1,l]-2*X[j,l]+X[j-1,l])
                     X_hat_2[i,l] += X[j,i]*(X[j+1,l]-X[j-1,l])
                end
            end
        end
                
        rhs_L = -1/(2*dt)*X_hat*L'+0.5*X_hat_2*LA/dx;


        rhs_L = -rhs_L .- L'*(obj.sigmaT*I - obj.settings.sigmaS*G); 

        L .= L .+ obj.settings.dt*rhs_L';
                
        WNew,STmp = qr(L);
        WNew = WNew[:,1:r];

        NUp = WNew' * W;
        W .= WNew;
        X .= XNew;

        ################## S-step ##################
        S .= MUp*S*(NUp')
        SWAW=S*W'*obj.A*W;
        WGW = W'*G*W;

        X_hat = zeros(obj.settings.r,obj.settings.r)
        X_hat_2 = zeros(obj.settings.r,obj.settings.r)
        
        for i=1:obj.settings.r
            for l=1:obj.settings.r
                for j=2:(obj.settings.NCells-1)
                     X_hat[i,l] += X[j,i]*(X[j+1,l]-2*X[j,l]+X[j-1,l])
                     X_hat_2[i,l] += X[j,i]*(X[j+1,l]-X[j-1,l])
                end
            end
        end

        rhs_S = -1/(2*dt)*X_hat*S+0.5*X_hat_2*SWAW/dx;

        rhs_S = -rhs_S .- S*obj.sigmaT + obj.settings.sigmaS*S*WGW; 
        
        S .= S .+ obj.settings.dt.*rhs_S;

        # S .= S .+ obj.settings.dt.*X'*F(obj,X*S*W',n)*W;
        
        t = t+obj.settings.dt;
    end

    # return end time and solution
    return t, X,S,W;

end
