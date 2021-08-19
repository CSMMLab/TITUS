include("settings.jl")
include("Solver.jl")
using LinearAlgebra

function SolveNaiveUnconventional(obj::Solver)
    # Get rank
    r=obj.settings.r;
    
    # Set up initial condition
    u = SetupIC(obj::Solver);

    # Low-rank approx of init data:
    X,S,W = svd(u');
    
    # rank-r truncation:
    X = X[:,1:r];
    W = W[:,1:r];
    S = Diagonal(S);
    S = S[1:r, 1:r];

    for n = 1:Nt

        ################## K-step ##################
        K .= X*S;

        K .= K .+ dt*Rhs(obj,K*W')*W;

        XNew,STmp = qr(K);

        MUp = XNew' * X;

        ################## L-step ##################
        L = W*S';

        L .= L .+ dt*(X'*F(obj,X*L'))';
                
        WNew,STmp = qr(L);

        NUp = WNew' * W;
        W .= WNew;
        X .= XNew;

        ################## S-step ##################
        S .= MUp*S*(NUp')

        S .= S .+ dt.*X'*Rhs(obj,X*S*W')*W;
        
        t = t+dt;
    end

    # return end time and solution
    return t, X,S,W;

end

