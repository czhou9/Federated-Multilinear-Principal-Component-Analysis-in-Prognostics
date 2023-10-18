index_FMPCA = randperm(284);
%X_4D = TC;
X_4D = X_4D(:,:,:,index_FMPCA);
Ym_t = Ym_t(index_FMPCA);


I1 = size(X_4D,1);
I2 = size(X_4D,2);
I3 = size(X_4D,3);
N_user1 = 140;
N_user2 = 57;
N_user3 = 30;
N_train = 227;
N_test = 57;
N_total = N_train + N_test;

%% Initialize U1,U2,U3 based on MPCA
% X_4D = TC;
% N_train = 160;
% N_test = 40;
% N_total = N_train + N_test;
X_MPCA = X_4D(:,:,:,1:N_train);
%X_MPCA = TC(:,:,:,1:N);
Ym_MPCA = Ym_t(1:N_train);

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
S_U4MPCA = [ones(N_train,1) double(tenmat(S_MPCA,4))];
Beta_MPCA = pinv(S_U4MPCA' * S_U4MPCA) * S_U4MPCA' * log(Ym_MPCA);
Beta1_MPCA = Beta_MPCA(2:end,:);
Beta0_MPCA = Beta_MPCA(1,:);


%% Estimate TTF 
% Select last 20% data as test set
Ym_testMPCA = Ym_t((N_train + 1): N_total);
% Generate S matrix of the test dataset 
S_testMPCA = double(ttm(tensor(X_4D(:,:,:,(N_train + 1): N_total)), {U1_MPCA, U2_MPCA, U3_MPCA}, [1 2 3]));
%S_testMPCA = double(ttm(tensor(TC(:,:,:,((N+1):(1.25*N)))), {U1_MPCA, U2_MPCA, U3_MPCA}, [1 2 3]));
% Generate estimated Ym and make prediction error
Ym_estMPCA = exp(Beta0_MPCA * ones(N_test,1)  + double(tenmat(S_testMPCA,4)) * Beta1_MPCA);
PredEr_MPCA = abs(Ym_testMPCA-Ym_estMPCA)./ Ym_testMPCA;
%boxplot(PredEr_MPCA)

%save('MPCA_NumericalStudy.mat','PredEr_MPCA')
save('MPCA_CaseStudy.mat','PredEr_MPCA')

%  PredEr_MPCA = [PredEr_MPCA1 ; PredEr_MPCA2 ; PredEr_MPCA3 ; PredEr_MPCA4 ; PredEr_MPCA5 ;...
%              PredEr_MPCA6 ; PredEr_MPCA7 ; PredEr_MPCA8 ; PredEr_MPCA9 ; PredEr_MPCA10 ];
%          