function [lambda mu eta beta gamma] = kf_grad ( ...
    Fp1, H, Sc, Si, p_S, p_z, z, Pp, mup1, betap1)
% [lambda mu eta beta gamma] = kf_grad(F,Fp1,H,Sc,Si,p_S,p_z,z,Pp,mup1,betap1)
% Carry out one time step of computing the gradient of p(S) wrt to the
% hyperparameters.
%   p(S) is something like a log likelihood.
%   S = Sc'*Sc, where Sc is from kf_qrsc_update.
%   Si = inv(S). Call Si = kf_grad_Calc_invC(Sc) to get Si.
%   F, Fp1, H are the usual Kalman filter matrices. Fp1 is F at the next time
% index.
%   p_S is the partial derivative of p wrt S. You will likely use
%     Si = kf_grad_Calc_invC(Sc)
%     kf_grad_Calc_LogDetC_C(Si)
%     kf_grad_Calc_ztinvCz_C(z,Sc)
% to compute p_S, where z = y - H*xp. For example, if p is just the usual log
% likelihood so that, neglecting a constant term,
%     p = - 1/2 log(det(S)) - 1/2 z' inv(S) z,
% then
%     p_S = -0.5*kf_grad_Calc_LogDetC_C(Si) - 0.5*kf_grad_Calc_ztinvCz_C(Sc,z).
%   p_z is the partial derivative of p wrt z = y - H*xp. For p as above,
%     p_z = -Sc \ (z' / Sc)' % = inv(S) z.
%   mup1, betap1 are mu and beta from step k+1, where this is currently step
% k. Set mu = zeros(N^2,1) and beta = zeros(N,1) for the first call to this
% function.
%   Pp is the prediction-step covariance matrix.
%   The outputs lambda and mu are used to calculate the gradient. The total
% derivative wrt hyperparameter a is
%     sum_{k = 1}^N -lambda_k' R_a(:) - mu_k' Q_a(:),
% where R_a, Q_a are partial derivatives of R and Q. In many cases, R_a and Q_a
% are almost entirely zero and the nonzero structure is known; it's important to
% take advantage of these facts in order to make the gradient computation
% efficient.

% AMB ambrad@cs.stanford.edu
% CDFM, Geophysics, Stanford
  
  [no ns] = size(H);
  vec = @(x) x(:);
  % Solve
  %     eta_k' f_{Pf_k} + mu_{k+1}' g_{Pf_k} = 0
  % for eta.
  eta = Calc_vt_ABAt_B(mup1, Fp1); % -mu_{k+1}' g_Pf
  % Not needed because Calc_vt_ABAt_B takes care of it implicitly:
  %    eta = Sym(eta);
  % Solve
  %     beta_{k+1}' (b_{k+1})_{xf_k} + gamma_k' c_{xf_k} = 0
  % for gamma.
  gamma = betap1(:)'*Fp1; % -beta_{k+1}' b_xf
  % Solve
  %     p_{S_k} + lambda_k' s_{S_k} + eta_k' f_{S_k} + gamma_k' c_{S_k} = 0
  % for lambda.
  lambda = -p_S(:)' +...
           -Calc_vt_AinvBAt_B(eta, Pp*H', Si) +...     % -eta' f_S
           vec(Calc_vtinvAb_A((gamma*Pp)*H', Si, z))'; % -gamma' c_S
  lambda = Sym(lambda);
  % Solve
  %     lambda_k' s_{Pp_k} + mu_k' g_{Pp_k} + eta_k' f_{Pp_k} +
  %       gamma_k' c_{Pp_k} = 0
  % for mu.
  V = Sc' \ H;
  Siz = Sc \ (z' / Sc)';
  mu = eta - Calc_vt_ABAt_A(eta, Pp, V'*V) +... % -eta' f_Pp
       Calc_vt_ABAt_B(lambda, H) +...           % -lambda' s_Pp
       vec(Calc_vtAb_A(gamma, H'*Siz))';        % -gamma c_Pp
  mu = Sym(mu);
  % Solve
  %     p_{xf_k} + beta_k' b_{xp_k} + gamma_k' c_{xp_k} = 0
  % for beta.
  beta = gamma - ((gamma*Pp)*V')*V +... % -gamma' c_xp
         p_z(:)'*H;                     % -p_z z_xp
end

function p = Calc_vt_kronAA (v, A, vissym)
% For v = v' if vissym, general v otherwise, compute 
%     p = v(:)'*kron(A,A);
% efficiently. This straightforward version is O(N^2 M^2) for [M N] =
% size(A). The efficient version that follows is O(m^2 n + n^2 m).  
  [m n] = size(A);
  if (vissym)
    v = reshape(v, m, m);
    v = v - diag(diag(v))/2;
    p = A'*triu(v)*A;
    p = p + p';
  else
    p = A'*(reshape(v, m, m)*A);
  end
  p = p(:)';
end

function v = Calc_vt_invC_C (v, Ci)
% For C = R' R, return
%     z(:)' inv(C)_C.
  v = -Calc_vt_kronAA(v, Ci, false);
end
  
function v = Calc_vt_ABAt_B (v, A)
% For B = B', v = v', return
%     v(:)' (A B A')_B.
  v = Calc_vt_kronAA(v(:), A, true);
end
  
function v = Calc_vtAb_A (v, b)
% Return (v' A b)_A.
  v = v(:)*b(:)';
end
  
function v = Calc_vtinvAb_A (v, Ai, b)
% Return (v' inv(A) b)_A for A = A'.
  n = size(Ai, 1);
  v = Calc_vtAb_A(v(:), b);
  v = reshape(Calc_vt_invC_C(v(:), Ai), n, n);
end
  
function v = Calc_vt_AinvBAt_B (v, A, Bi)
% For B = B', v = v', Bi = inv(B), return
%   v(:)' (A inv(B) A')_B.
  
  % v' (A inv(B) A')_B = v' [(A inv(B) A')_inv(B)] [inv(B)_B]
  v = Calc_vt_kronAA(v, A, true);
  v = -Calc_vt_kronAA(v, Bi, true);
end

function p = Calc_vt_ABAt_A (v, A, B)
% For B = B', v = v', return
%     v(:)' (A B A')_A.
% This is an O(m^2 n + m n^2) operation for [m n] = size(A).
  m = size(A, 1);
  v = reshape(v, m, m);
  % If v were unsymmetric, we would use this:
  %     p = (v' + v)*A*B;
  p = 2*v*A*B;
  p = p(:)';
end

function A = Sym (A)
% Return (A + A')/2.
  n = sqrt(numel(A));
  A = reshape(A, n, n);
  A = (A + A')/2;
  A = A(:)';
end
