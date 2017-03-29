function [x Psc] = kf_chol_smooth(F,xf,xsp1,Ppc,Pfc,Psp1c)
% [x Psc] = kf_chol_smooth(F,xf,xsp1,Ppc,Pfc,Psp1c)
% RTS smoother.
%   On input
%     F is the state transition matrix.
%     xf = x_k|k
%     xsp1 = x_k+1|n
%     Ppc = chol(P_k+1|k)
%     Pfc = chol(P_k|k)
%     Psp1c = chol(P_k+1|n).
%   On output,
%     x = x_k|n
%     Psc = chol(P_k|n).
%   The RTS smoother update is
%     1. S_k = P_k|k F' inv(P_k+1|k)
%     2. x_k|n = x_k|k + S_k (x_k+1|n - x_k+1|k)
%     3. P_k|n = P_k|k + S_k (P_k+1|n - P_k+1|k) S_k'
% This version of the smoother update is mostly numerically safe. It's 2-4x
% faster than kf_qrsc_smooth.
  
% AMB ambrad@cs.stanford.edu
% CDFM, Geophysics, Stanford
  
  n = numel(xf);
  Pf = Pfc'*Pfc;
  PfFt = Pf*F';
  clear Pfc;
  
  A = PfFt*(Ppc \ (Psp1c / Ppc)');
  B = PfFt*inv(Ppc);
  Psc = chol(Pf + A*A' - B*B');
  
  x = xf + PfFt*(Ppc \ ((xsp1 - F*xf)' / Ppc)');
end
