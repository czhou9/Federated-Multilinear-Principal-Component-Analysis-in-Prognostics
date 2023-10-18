%% Initialize U1,U2,U3 based on MPCA
X_MPCA = X_4D(:,:,:,1:40);
%X_MPCA = TC(:,:,:,1:N);
Ym_MPCA = Ym_t(1:40);

TX = X_MPCA;
gndTX = Ym_MPCA;
testQ = 97;
maxK = 1;
[tUs, odrIdx, TXmean, Wgt]  = MPCA(TX,gndTX,testQ,maxK);

U1_MPCA = tUs{1,1};
U2_MPCA = tUs{2,1};
U3_MPCA = tUs{3,1};


%% Derive Beta0 and Beta1
S_MPCA = double(ttm(tensor(X_MPCA), {U1_MPCA, U2_MPCA, U3_MPCA}, [1 2 3]));
S_U4MPCA = [ones(N,1) double(tenmat(S_MPCA,4))];
Beta_MPCA = pinv(S_U4MPCA' * S_U4MPCA) * S_U4MPCA' * log(Ym_MPCA);
Beta1_MPCA = Beta_MPCA(2:end,:);
Beta0_MPCA = Beta_MPCA(1,:);

%% Estimate TTF 
% Select last 20% data as test set
Ym_testMPCA = Ym_t((N+1): (1.25*N));
% Generate S matrix of the test dataset 
%S_testMPCA = double(ttm(tensor(TC(:,:,:,((N+1):(1.25*N)))), {U1_MPCA, U2_MPCA, U3_MPCA}, [1 2 3]));
S_testMPCA = double(ttm(tensor(X_4D(:,:,:,((N+1):(1.25*N)))), {U1_MPCA, U2_MPCA, U3_MPCA}, [1 2 3]));
% Generate estimated Ym and make prediction error
Ym_estMPCA = exp(Beta0_MPCA * ones(0.25*N,1)  + double(tenmat(S_testMPCA,4)) * Beta1_MPCA);
PredEr_MPCA = abs(Ym_testMPCA-Ym_estMPCA)./ Ym_testMPCA;
%save('Benchmark_MPCA_Complete.mat','PredEr_MPCA')
%save('Benchmark_MPCA_Miss10%.mat','PredEr_MPCA')
%PredEr_MPCA_cv = PredEr_MPCA;
%save('Benchmark_MPCA_cv_50%.mat','PredEr_MPCA_cv');


%PredEr_MPCA(PredEr_MPCA(:)>1)=[];
%% Compare MPCA and new proposed method from boxplot
%boxplot([PredEr_MPCA,PredEr_alpha,PredEr_cv],'Notch','on','Labels',{'MPCA','Proposed_RankInMPCA','Proposed_RankInCV'}) 
%boxplot(PredEr_MPCA)
%boxplot(PredEr_alpha(:,2))