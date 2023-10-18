function [U1]=incrementalSVD_localoptimization1(Signal,N_user1,P2,P3)


N=size(Signal,2);
%%%%%%%%%%Initinalization Using the First User's Data%%%%%%%%%%%%%%%%%%%%%%%%%
Ut=null(1);St=null(1);Vt=null(1);
FirstUser=Signal(:,1:(N_user1*P2*P3));%%%%%%%%%%%%%%
r = rank(FirstUser);%%%%%%%%%%%%%%
[Ut, St, Vt]=svd(FirstUser);%%%%%%%%%%%%%%%
Ut = Ut(:,1:r);%%%%%%%%%%%%%
St = St(1:r,1:r);%%%%%%%%%%%%%%%%
Vt = Vt(:,1:r);%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Signal=Signal(:,(N_user1*P2*P3+1):N);%%%%%%%%%%%%%%
for i=1:(N-N_user1*P2*P3)
    vt=Signal(:,i);
    %if i==1%%%%%%%%%%%%%%
        %rt=vt;%%%%%%%%%%%%%%
        %wt=null(1);%%%%%%%%%%%%%%
    %else%%%%%%%%%%%%%%
        wt=Ut'*vt;
        pt=Ut*wt;
        rt=vt-pt;
    %end%%%%%%%%%%%%%%
    
    if isempty(St)==1
        r=0;
        c=0;
    else
        [r, c]=size(St);
    end
    

    TempMat=[St wt;zeros(1,c) norm(rt)];
    [Uhat, Shat, Vhat]=svd(TempMat);

    Ut=[Ut rt/norm(rt)]*Uhat;
    St=Shat;
    
    if isempty(Vt)==1
        r=0;
        c=0;
    else
        [r, c]=size(Vt);
    end
    Vt=[Vt zeros(r,1);zeros(1,c) 1]*Vhat;
    if (norm(rt))<1e-6
        Ut=Ut(:,1:end-1);
        St=St(1:end-1,1:end-1);
        Vt=Vt(:,1:end-1);
    end

end
U1 = Ut;
end