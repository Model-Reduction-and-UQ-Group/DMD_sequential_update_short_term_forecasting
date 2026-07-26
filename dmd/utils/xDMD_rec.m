function [u_xdmd]=xDMD_rec(AA,U,u,m)
%uso matrice A calibrata con xDMD per ricostruire intera serie:
%dove ho dato vero disponibile lo uso per calcolare il dato al tempo successivo
%dove ho un nan, uso la previsione ottenuta allo step precedente per calcolare il successivo

    u_xdmd = zeros(size(u,1),m);
    u_xdmd(:,1) = u(:,1);

for i=1:m-1
  if isnan(u(:,i))
      u0=u_xdmd(:,i);      %dove no dato uso la previsione del DMD
  else
      u0=u(:,i);      %altrimenti se ho dato uso quello noto
%       u0=u_xdmd(:,i);        %---------------prova a fare tutto in extr
  end
  
  aa=U'*[u0;1];
  u_xdmd(:,i+1)=u0+AA*aa;
end




end