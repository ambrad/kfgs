function [x Pc] = kf_qrsc_smooth(F,Qc,xf,xsp1,Ppc,Pfc,Psp1c)
% function [x Pc] = kf_qrsc_smooth(F,Qc,xf,xsp1,Ppc,Pfc,Psp1c)
% RTS smoother.
%   On input
%     F is the state transition matrix.
%     Qc = chol(Q), where Q is the process covariance matrix.
%     xf = x_k|k
%     xsp1 = x_k+1|n
%     Ppc = chol(P_k+1|k)
%     Pfc = chol(P_k|k)
%     Psp1c = chol(P_k+1|n).
%   On output,
%     x = x_k|n
%     Pc = chol(P_k|n).
%   The RTS smoother update is
%     1. S_k = P_k|k F' inv(P_k+1|k)
%     2. x_k|n = x_k|k + S_k (x_k+1|n - x_k+1|k)
%     3. P_k|n = P_k|k + S_k (P_k+1|n - P_k+1|k) S_k'
% This version of the smoother update is completely numerically safe.
  n = size(F,1);
  % C C' = S_k P_k+1|n S_k'
  C = Pfc'*(Pfc*(F'*(Ppc \ (Psp1c / Ppc)')));  
  % A'A = [P_k+1|k   F P_k|k
  %        P_k|k F'  P_k|k + S_k P_k+1|n S_k'],
  % and the Schur complement of P_k+1|k in A'A is P_k|n. Just as in the Kalman
  % update, qr(A) is safe in the presence of numerical error and so assures a
  % factorization of P_k|n.
  Z = zeros(n,n);
  A = [Pfc*F' Pfc; full(Qc) Z; Z C'];
  [~,A] = qr(A,0);
  % Extract the factorization of the Schur complement
  Pc = A(n+1:end,n+1:end);
  mask = diag(Pc < 0);
  Pc(mask,:) = -Pc(mask,:);
  % Update state
  x = xf + Pfc'*(Pfc*(F'*(Ppc \ ((xsp1 - F*xf)' / Ppc)')));
end
