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

