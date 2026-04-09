function [X1,X2,X1_tilde,Y]=buildXcouple(Train)

%type: X1 and X2 shifted
X1 = Train(:,:,1);
X2 = Train(:,:,2);
Y=X2-X1;
X1_tilde = [X1;ones(1,size(X1,2))];

end