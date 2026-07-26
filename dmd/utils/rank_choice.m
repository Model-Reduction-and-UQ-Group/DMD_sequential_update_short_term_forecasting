function [rvec,r]=rank_choice(X1,Y,X1_tilde, type)

%type:  1=rank;  2=r90;  3=cut off 10^-3; 4=cut off 10^-5; 5=cut off 10^-8; 6=gavish;


%% svd
[U, S, V]=svd(X1,'econ');
[Utilde, Stilde, Vtilde]=svd(X1_tilde,'econ');
% [Uhat, Shat, Vhat]=svd(Y,'econ');

% save(sprintf('autovalori_%s.mat',IoRoE), 'S', 'Stilde');%, 'Shat')
% plot(diag(S))


%% truncation rank options

if type==1          %rank
    
    r1=rank(X1);
    r2=rank(X1_tilde);
%     r3=rank(Y);

    
elseif  type==2           %r90

    trunc_en=0.90;

    cdS1 = cumsum(diag(S))/sum(diag(S));  % Cumulative energy
    r1 = min(find(cdS1>trunc_en));            % Find r to capture 90% energy

    cdS2 = cumsum(diag(Stilde))/sum(diag(Stilde));  % Cumulative energy
    r2 = min(find(cdS2>trunc_en));            % Find r to capture 90% energy

%     cdS3 = cumsum(diag(Shat))/sum(diag(Shat));  % Cumulative energy
%     r3 = min(find(cdS3>trunc_en))            % Find r to capture 90% energy

elseif  type==3         %r* (10^-3)
    
    trunc=10^-2;

    index1 = find(diag(S)<= sum(diag(S))*trunc);
    r1 = min(index1);

    index2 = find(diag(Stilde)<= sum(diag(Stilde))*trunc);
    r2 = min(index2);

    % index3 = find(diag(Shat)<= trunc);
    % r3 = min(index3);
elseif  type==4       %r* (10^-5)
    
    trunc=10^-2
    
    index1 = find(diag(S)<= sum(diag(S))*trunc);
    r1 = min(index1);
    
    index2 = find(diag(Stilde)<= sum(diag(Stilde))*trunc);
    r2 = min(index2);
    
%     index3 = find(diag(Shat)<= sum(diag(Shat))*trunc);
%     r3 = min(index3)

elseif  type==5       %r* (10^-8)
    
    trunc=10^-2;
    
    index1 = find(diag(S)<= sum(diag(S))*trunc);
    r1 = min(index1);
    
    index2 = find(diag(Stilde)<= sum(diag(Stilde))*trunc);
    r2 = min(index2);
    
%     index3 = find(diag(Shat)<= sum(diag(Shat))*trunc);
%     r3 = min(index3)

elseif  type==6         % gavish

    beta=size(X1,2)/size(X1,1);
    omega=0.56*beta^3-0.95*beta^2+1.82*beta+1.43;
    index1 = find(diag(S)< (omega*median(diag(S))));
    r1 = min(index1);

    beta=size(X1,2)/size(X1,1);
    omega=0.56*beta^3-0.95*beta^2+1.82*beta+1.43;
    index2 = find(diag(Stilde)< (omega*median(diag(Stilde))));
    r2 = min(index2);
   
    % beta=size(X1,2)/size(X1,1);
    % omega=0.56*beta^3-0.95*beta^2+1.82*beta+1.43;
    % index3 = find(diag(Shat)< (omega*median(diag(Shat))));
    % r3 = min(index3);
    
end

r3=NaN;

rvec=[r1,r2,r3];
r=max([r1,r2,r3]);

end
