function [loglik g] = kf_loglik_grad (rcd, fn, Ns, Nt, ofn)
% [loglik g] = kf_loglik_grad(rcd,fn,Ns,Nt,ofn)
%   Compute the log-likelihood of the observations and its gradient.
%
% This version does not use checkpointing. See kf_loglik_grad_cp for a
% description of the inputs to this routine.

% AMB ambrad@cs.stanford.edu
% CDFM, Dept of Geophysics, Stanford
  
  hofn = nargin > 5;
  if (~hofn) ofn = []; end
  
  % Forward.
  loglik = 0;
  for (it = 1:Nt)
    % Predict.
    if (it > 1)
      [F Q] = fn('fq',it);
      [xp Ppc] = kf_qrsc_predict(F, Q, xf, Pfc);
    else
      [xp Ppc] = fn('i');
    end
    % Update.
    [H Rc y] = fn('hry', it);
    [xf Pfc z Sc] = kf_qrsc_update(H, Rc, y, xp, Ppc);
    % Contribution to loglik.
    loglik = loglik + fn('ll', it, Sc, z);
    if (hofn) ofn(it, xp, Ppc, xf, Pfc, z, Sc); end
    % Record (for kfgs internal use).
    if (it < Nt)
      rcd = kf_rcd('push', rcd, Ppc, Sc, z);
    end
  end

  % Adjoint.
  mu = zeros(Ns^2, 1);
  beta = zeros(Ns, 1);
  opts.UT = true;
  for (it = Nt:-1:1)
    if (it < Nt)
      [rcd Ppc Sc z] = kf_rcd('pop', rcd);
    end
    Pp = Ppc'*Ppc;
    Si = linsolve(full(Sc), eye(size(Sc, 1)), opts); Si = Si*Si'; % inv(S)
    [p_S p_z] = fn('p', it, Sc, Si, z);
    if (it == Nt) Fp1 = sparse(Ns, Ns); else Fp1 = fn('f', it+1); end
    H = fn('h', it);
    [lambda mu eta beta gamma] = kf_grad( ...
      Fp1, H, Sc, Si, p_S, p_z, z, Pp, mu, beta);
    git = fn('g', it, lambda, mu, eta, beta, gamma);
    if (it == Nt) g = git; else g = g + git; end
  end
end
