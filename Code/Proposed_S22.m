% index_FMPCA = randperm(250);
% X_4D = TC;
% X_4D = X_4D(:,:,:,index_FMPCA);
% Ym_t = Ym_t(index_FMPCA);
% save('X_4D_Ym_t_CaseStudy_sample2.mat','X_4D','Ym_t')
tic
%% Set user, train and test sample for numerical study
P1 = 2;
P2 = 2;
P3 = 2;
I1 = size(X_4D,1);
I2 = size(X_4D,2);
I3 = size(X_4D,3);
N_user1 = 50;
N_user2 = 20;
N_user3 = 10;
N_train = 80;
N_test = 20;
N_total = N_train + N_test;
X_bar = zeros(I1, I2, I3, N_train);
for i = 1:N_train
    X_bar(:,:,:,i) = X_4D(:,:,:,i) - mean(X_4D(:,:,:,1:N_train),4);
end
X_user1 = X_bar(:,:,:,1:N_user1);
X_user2 = X_bar(:,:,:,(N_user1 + 1):(N_user1 + N_user2));
X_user3 = X_bar(:,:,:,(N_user1 + N_user2 + 1):(N_user1 + N_user2 + N_user3));
X_train = X_bar(:,:,:,1:N_train);
X_train_NCenter = X_4D(:,:,:,1:N_train);
X_test_NCenter = X_4D(:,:,:,(N_train + 1):N_total);
Ym_user1 = Ym_t(1:N_user1);
Ym_user2 = Ym_t((N_user1 + 1):(N_user1 + N_user2));
Ym_user3 = Ym_t((N_user1 + N_user2 + 1):(N_user1 + N_user2 + N_user3));
Ym_train = Ym_t(1:N_train);
Ym_test = Ym_t((N_train + 1):N_total);

%% Set user, train and test sample for case study
 %X_4D = TC;
% P1 = 3;
% P2 = 3;
% P3 = 3;
% I1 = size(X_4D,1);
% I2 = size(X_4D,2);
% I3 = size(X_4D,3);
% N_user1 = 50;
% N_user2 = 20;
% N_user3 = 10;
% N_train = 80;
% N_test = 20;
% N_total = N_train + N_test;
% X_bar = zeros(I1, I2, I3, N_train);
% for i = 1:N_train
%     X_bar(:,:,:,i) = X_4D(:,:,:,i) - mean(X_4D(:,:,:,1:N_train),4);
% end
% X_user1 = X_bar(:,:,:,1:N_user1);
% X_user2 = X_bar(:,:,:,(N_user1 + 1):(N_user1 + N_user2));
% X_user3 = X_bar(:,:,:,(N_user1 + N_user2 + 1):(N_user1 + N_user2 + N_user3));
% X_train = X_bar(:,:,:,1:N_train);
% X_train_NCenter = X_4D(:,:,:,1:N_train);
% X_test_NCenter = X_4D(:,:,:,(N_train + 1):N_total);
% Ym_user1 = Ym_t(1:N_user1);
% Ym_user2 = Ym_t((N_user1 + 1):(N_user1 + N_user2));
% Ym_user3 = Ym_t((N_user1 + N_user2 + 1):(N_user1 + N_user2 + N_user3));
% Ym_train = Ym_t(1:N_train);
% Ym_test = Ym_t((N_train + 1):N_total);

%% Initialization
X_train_un1 = double(tenmat(tensor(X_train),1));
X_train_un2 = double(tenmat(tensor(X_train),2));
X_train_un3 = double(tenmat(tensor(X_train),3));
U1 = incrementalSVD_initialization1(X_train_un1,N_user1,I2,I3);
U2 = incrementalSVD_initialization2(X_train_un2,N_user1,I1,I3);
U3 = incrementalSVD_initialization3(X_train_un3,N_user1,I1,I2);

% U1 = U1(:,1:P1);
% U2 = U2(:,1:P2);
% U3 = U3(:,1:P3);



%% Local Optimization
tic
%Set initial iteration criterion
%S_train_0 = double(ttm(tensor(X_train), {U1, U2, U3}, [1 2 3]));
S_train_0 = double(ttm(tensor(X_train), {U1', U2', U3'}, [1 2 3]));

S_train_fro_0 = power(norm(double(tenmat(tensor(S_train_0),4)),'fro'),2);
ita = 100;
iteration = 0;
%while  (ita > 10^(-5)) && (iteration < 20)
while  (abs(ita) > 10^(-6)) && (iteration < 1)
    %U1 in local optimization
    %S_train_1 = double(ttm(tensor(X_train), { U2, U3}, [2 3]));
    S_train_1 = double(ttm(tensor(X_train), { U2', U3'}, [2 3]));
    S_train_1_un1 = double(tenmat(tensor(S_train_1),1));
    %U1 = incrementalSVD_localoptimization1(S_train_1_un1,N_user1,P2,P3);
    U1 = incrementalSVD_initialization1(S_train_1_un1,N_user1,I2,I3);
    %U1 = U1(1:P1,:);
    %U1 = U1(:,1:P1);
    
    %U2 in local optimization
    %S_train_2 = double(ttm(tensor(X_train), { U1, U3}, [1 3]));
    S_train_2 = double(ttm(tensor(X_train), { U1', U3'}, [1 3]));
    S_train_2_un2 = double(tenmat(tensor(S_train_2),2));
    %U2 = incrementalSVD_localoptimization2(S_train_2_un2,N_user1,P1,P3);
    U2 = incrementalSVD_initialization2(S_train_2_un2,N_user1,I1,I3);
    %U2 = U2(1:P2,:);
    %U2 = U2(:,1:P2);
    
    %U3 in local optimization
    %S_train_3 = double(ttm(tensor(X_train), { U1, U2}, [1 2]));
    S_train_3 = double(ttm(tensor(X_train), { U1', U2'}, [1 2]));
    S_train_3_un3 = double(tenmat(tensor(S_train_3),3));
    %U3 = incrementalSVD_localoptimization3(S_train_3_un3,N_user1,P1,P2);
    U3 = incrementalSVD_initialization3(S_train_3_un3,N_user1,I1,I2);
    %U3 = U3(:,1:P3);
    %U3 = U3(1:P3,:);
    
    
    
    %Calculate the frobenius norm of low-dimensional tensor 
    %S_train = double(ttm(tensor(X_train), {U1, U2, U3}, [1 2 3]));
    S_train = double(ttm(tensor(X_train), {U1', U2', U3'}, [1 2 3]));
    S_train_fro = power(norm(double(tenmat(tensor(S_train),4)),'fro'),2);
    
    ita = S_train_fro - S_train_fro_0;
    S_train_fro_0  = S_train_fro;
    iteration = iteration + 1;
end
toc

 U1 = U1(:,1:P1);
 U2 = U2(:,1:P2);
 U3 = U3(:,1:P3);



%% Derive Beta0 and Beta1
%S_FMPCA = double(ttm(tensor(X_train), {U1, U2, U3}, [1 2 3]));
S_FMPCA = double(ttm(tensor(X_train_NCenter), {U1', U2', U3'}, [1 2 3]));

S_U4FMPCA = [ones(N_train,1) double(tenmat(S_FMPCA,4))];
Beta_FMPCA = pinv(S_U4FMPCA' * S_U4FMPCA) * S_U4FMPCA' * log(Ym_train);
Beta1_FMPCA = Beta_FMPCA(2:end,:);
Beta0_FMPCA = Beta_FMPCA(1,:);


%% Estimate TTF 

% Generate S matrix of the test dataset 
%S_testFMPCA = double(ttm(tensor(X_test), {U1, U2, U3}, [1 2 3]));
S_testFMPCA = double(ttm(tensor(X_test_NCenter), {U1', U2', U3'}, [1 2 3]));

%S_testMPCA = double(ttm(tensor(TC(:,:,:,((N+1):(1.25*N)))), {U1_MPCA, U2_MPCA, U3_MPCA}, [1 2 3]));

% Generate estimated Ym and make prediction error
Ym_estFMPCA = exp(Beta0_FMPCA * ones(N_test,1)  + double(tenmat(S_testFMPCA,4)) * Beta1_FMPCA);
PredEr_FMPCA = abs(Ym_test-Ym_estFMPCA)./ Ym_test;

boxplot(PredEr_FMPCA)
toc
%save('FMPCA_NumericalStudy_sample2.mat','PredEr_FMPCA')
%save('FMPCA_NumericalStudy_noise0.1.mat','PredEr_FMPCA')