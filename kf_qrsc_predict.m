function [xp Ppc] = kf_qrsc_predict(F,Q,xf,Pfc)
% [xp Ppc] = kf_qrsc_predict(F,Q,xf,Pfc)
% On input,
%   F is the state transition matrix.
%   Q is the process covariance.
%   xf = x_k-1|k-1
%   Pfc = chol(P_k-1|k-1).
% On output,
%   xp = x_k|k-1
%   Ppc = chol(P_k|k-1).
  
% In all the kf_qrsc_*.m routines, I use the following notation:
%
% Predict:
%   xp = F xf
%   Pp = F Pf F' + Q
% Update (aka filter):
%   z = y - H*xp
%   S = H Pp H' + R
%   K = Pp H' inv(S)
%   xf = xp + K z
%   Pf = (I - K H) Pp
%
% Kc = chol(K)

% AMB ambrad@cs.stanford.edu
% CDFM, Geophysics, Stanford
  
  xp = F*xf;
  PfcFp = Pfc*F';
  % PfcFp'*PfcFp is assuredly p.d., so a simple chol works.
  Pp = PfcFp'*PfcFp + Q;
  Ppc = chol(Pp);
end
