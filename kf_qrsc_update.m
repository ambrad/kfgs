function [xf Pfc z Sc] = kf_qrsc_update(H,Rc,y,xp,Ppc)
% [xf Pfc z Sc] = kf_qrsc_update(H,Rc,y,xp,Ppc)
% On input,
%   H is the state -> observation matrix.
%   Rc is chol(R), where R is the observation covariance matrix.
%   y is the vector of observations.
%   xp = x_k|k-1
%   Ppc = chol(P_k|k-1).
% On output,
%   xf = x_k|k
%   Pfc = chol(P_k|k)
%   z = y - H*xp
%   Sc = chol(H P_k|k-1 H' + R) [if requested].
%
% z and Sc can be used to calculate the likelihood
%     p(y_k | Y_k-1),
% where Y_k-1(:,i) is measurement vector i, as follows. First,
%     p(y_k | Y_k-1) = N(y_k; H x, R + H P_k|k-1 H')
%                    = N(y_k - Hx; 0, Sc*Sc')
% where N(x; mu, Sigma) is the density for a normal distribution having mean mu
% and covariance Sigma. Second, if R = chol(A), then
%     log(det(A)) = 2*sum(log(diag(R))).
% Hence
%     log p(y_k | Y_k-1) = -n/2 log(2 pi) - sum(log(diag(Sc))) - 1/2 q q',
% where q = z' / Sc (or q = (Sc' \ z)'), n = length(y_k), and I have just
% taken the logarithm of the expression for the normal probability density
% function.
  
% AMB ambrad@cs.stanford.edu
% CDFM, Geophysics, Stanford
  
  [n m] = size(H);
  % For
  %     A = [chol(R)           0
  %          chol(P_k|k-1) H'  chol(P_k|k-1)],
  % the Schur complement of (A'A)(1:ny,1:ny) in A'A is
  %     P_k|k = P_k|k-1 - P_k|k-1 H' inv(S) H P_k|k-1,
  % where
  %     S = H P_k|k-1 H' + R.
  %   The key idea in this square-root filter is that
  %     [~,R] = qr(A)
  % is a safe operation in the presence of numerical error and so assures a
  % factorization of the filtered covariance matrix P_k|k.
  A = [full(Rc) zeros(n,m); Ppc*H' Ppc];
  [~,Pfc] = qr(A,0);
  % If requested, extract chol(S).
  if (nargout > 3)
    Sc = Pfc(1:n,1:n);
    mask = diag(Sc < 0);
    Sc(mask,:) = -Sc(mask,:);
  end
  % Extract the part of chol(A) that is chol(P_k|k), ie, the
  % factorization of the Schur complement of interest.
  Pfc = Pfc(n+1:n+m,n+1:n+m);
  mask = diag(Pfc < 0);
  Pfc(mask,:) = -Pfc(mask,:);
  % Innovation
  z = y - H*xp;
  % Filtered state
  xf = xp + Pfc'*(Pfc*(H'*(Rc \ (Rc' \ z))));  
end
