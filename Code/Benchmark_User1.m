%% Initialize U1,U2,U3 based on MPCA
%X_4D = TC;
X_MPCA = X_4D(:,:,:,1:N_user1);
%X_MPCA = TC(:,:,:,1:N);
Ym_MPCA = Ym_t(1:N_user1);

TX = X_MPCA;
gndTX = Ym_MPCA;
testQ = 99.9;
maxK = 1;
[tUs, odrIdx, TXmean, Wgt]  = MPCA(TX,gndTX,testQ,maxK);

% p1 = 2;
% p2 = 2;
% p3 = 2;

p1 = 3;
p2 = 3;
p3 = 3;

U1_MPCA = tUs{1,1}(1:p1,:);
U2_MPCA = tUs{2,1}(1:p2,:);
U3_MPCA = tUs{3,1}(1:p3,:);

% U1_MPCA = tUs{1,1};
% U2_MPCA = tUs{2,1};
% U3_MPCA = tUs{3,1};


%% Derive Beta0 and Beta1
S_MPCA = double(ttm(tensor(X_MPCA), {U1_MPCA, U2_MPCA, U3_MPCA}, [1 2 3]));
S_U4MPCA = [ones(N_user1,1) double(tenmat(S_MPCA,4))];
Beta_MPCA = pinv(S_U4MPCA' * S_U4MPCA) * S_U4MPCA' * log(Ym_MPCA);
Beta1_MPCA = Beta_MPCA(2:end,:);
Beta0_MPCA = Beta_MPCA(1,:);

%% Estimate TTF 
% Select last 20% data as test set
Ym_testMPCA = Ym_t((N_train+1): N_total);
% Generate S matrix of the test dataset 
S_testMPCA = double(ttm(tensor(X_4D(:,:,:,((N_train+1): N_total))), {U1_MPCA, U2_MPCA, U3_MPCA}, [1 2 3]));
%S_testMPCA = double(ttm(tensor(TC(:,:,:,((N+1):(1.25*N)))), {U1_MPCA, U2_MPCA, U3_MPCA}, [1 2 3]));
% Generate estimated Ym and make prediction error
Ym_estMPCA = exp(Beta0_MPCA * ones(N_test,1)  + double(tenmat(S_testMPCA,4)) * Beta1_MPCA);
PredEr_MPCA_User1 = abs(Ym_testMPCA-Ym_estMPCA)./ Ym_testMPCA;
%boxplot(PredEr_MPCA_User1)

%save('User1_NumericalStudy.mat','PredEr_MPCA_User1')
save('User1_CaseStudy.mat','PredEr_MPCA_User1')

% PredEr_MPCA_User1 = [PredEr_MPCA_User11 ; PredEr_MPCA_User12 ; PredEr_MPCA_User13 ;...
%        PredEr_MPCA_User14 ; PredEr_MPCA_User15 ; PredEr_MPCA_User16 ; PredEr_MPCA_User17 ;...
%        PredEr_MPCA_User18 ; PredEr_MPCA_User19 ; PredEr_MPCA_User110];
   
   
