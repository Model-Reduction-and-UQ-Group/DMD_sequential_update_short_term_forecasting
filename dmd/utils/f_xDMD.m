function [AA,U] = f_xDMD(X1_tilde,X1,X2,trunc)
%% xDMD

    [U,Sigma,V] = svd(X1_tilde,'econ');
    k=trunc;
    U = U(:,1:k); Sigma = Sigma(1:k,1:k); V = V(:,1:k);    %silenzia per NO troncamento
%     A_xdmd = (X2-X1)*(V/Sigma*U');
    AA =(X2-X1)*V/Sigma;

end