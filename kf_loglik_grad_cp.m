function [loglik g] = kf_loglik_grad_cp (rcd, fn, Ns, Nt, Nbytes, ofn)
% [loglik g] = kf_loglik_grad_cp(rcd,fn,Ns,Nt,Nbytes,[ofn])
%   Compute the log-likelihood of the observations and its gradient.
%
%   This routine uses the adjoint method to compute the gradient in a time that
% is about the same as the time to compute the Kalman filter. It uses a mex
% interface to Andreas Griewank's checkpointing software REVOLVE to limit
% storage.
%
%   rcd is from kf_rcd('init').
%   Ns is the size of the state vector.
%   Nt is the number of time steps.
%   Nbytes is the number of bytes allowed in the buffer to compute the
% adjoint. For example, if you want to allow up to K steps' of data to be
% saved, set
%     Nbytes = K*8*(Ns+1)*Ns.
%   fn is a function handle that implements the following calls:
%     [x0 Pp0c] = fn('i'   )
%       Return the initial state and its covariance matrix.
%     F         = fn('f'  ,tidx)
%     [F Q]     = fn('fq' ,tidx)
%       Return the state transition matrix F and possibly the process noise Q
%       at time step 1 <= tidx <= Nt.
%     [H Rc y]  = fn('hry',tidx)
%     H         = fn('h',tidx)
%       Return the observation matrix H, the Cholesky factorization of the
%       observation noise R (chol(R)), and the observations y at time step
%       tidx.
%     p         = fn('ll', tidx,Sc,z)
%       Return the contribution of time step tidx to the log likelihood (or its
%       negative) given Sc = chol(R + H Pp H') and z = y - H xp. For example,
%       for the standard (negative of the) log-likelihood function, you would
%       write
%         [Sc z] = deal(varargin{1:2});
%         q = z' / Sc;
%         varargout{1} = sum(log(diag(Sc))) + 0.5*q*q';
%     [p_S p_z] = fn('p'  ,tidx,Sc,Si,z)
%       Return the partial derivative of the above function with respect to S
%       and z. For example, for the standard (negative of the) log-likelihood
%       function, you would write
%         [Sc Si z] = deal(varargin{1:3});
%         p_S = 0.5*(kf_grad_Calc_LogDetC_C(Si) + kf_grad_Calc_ztinvCz_C(z, Sc));
%         p_z = Sc \ (z' / Sc)';
%         varargout = {p_S p_z};
%     g         = fn('g'  ,tidx,lambda,mu,eta,beta,gamma)
%       Return the contribution of time step tidx to the gradient given the
%       Lagrange multipliers. Suppose the hyperparameter 'a' is in just Q and
%       R, and continuing the example of the negative log-likelihood above,
%       you need to compute:
%         [lambda mu] = deal(varargin{1:2});
%         % pseudocode:
%         varargout{1} = -lambda_k' R_a(:) - mu_k' Q_a(:)
%       The second line is pseudocode because the details of partial
%       derivative affect how the code is written. eta, beta, and gamma are
%       used if H or F have hyperparameters. See ex.m for an example.
%   ofn is an optional function handle that implements the following call:
%     ofn(it,xp,Ppc,xf,Pfc,z,Sc)
% It is called the first time a forward step is taken. The user can save
% desired output for step 'it'.
%
% See ex.m for an example of usage, and type 'help kf_grad' to read more
% about the mathematical operations necessary to compute in fn.
  
% AMB ambrad@cs.stanford.edu
% CDFM, Dept of Geophysics, Stanford

  %dbg = @(varargin)fprintf(1, varargin{:});
  dbg = @(varargin)1;
  hofn = nargin > 6;
  % Init stack.
  Nsnaps = min(Nt, floor(Nbytes/((Ns+1)*Ns*8)));
  % Init log likelihood.
  loglik = 0;
  % Init Lagrange multipliers.
  mu = zeros(Ns^2, 1);
  beta = zeros(Ns, 1);
  g = [];
  % Init REVOLVE parameters.
  capo = 1;
  check = -1;
  fine = Nt + capo;
  info = 0;
  done = false;
  opts.UT = true;
  while (~done)
    oldcapo = capo;
    [action check capo fine info] = revolve(check, capo, fine, Nsnaps, info);
    dbg('%10s %5d %5d\n', action, capo, check);
    switch (action)
      
     case 'takeshot'
      prev_check = check;
      if (capo ~= 1)
        rcd = kf_rcd('push', rcd, Pfc, xf);
      else
	[xp Ppc] = fn('i');
        rcd = kf_rcd('push', rcd, Ppc, xp);
      end
      
     case 'restore'
      if (check < prev_check)
        rcd = kf_rcd('discard', rcd);
      end
      prev_check = check;
      if (capo ~= 1)
        [rcd Pfc xf] = kf_rcd('peak', rcd);
      else
        [rcd Ppc xp] = kf_rcd('peak', rcd);
      end
      
     case 'advance'
      % Forward.
      dbg('      forward %d:%d\n', oldcapo, capo-1);
      for (it = oldcapo:capo-1)
	% Predict.
	if (it > 1)
	  [F Q] = fn('fq', it);
	  [xp Ppc] = kf_qrsc_predict(F, Q, xf, Pfc);
	end
	% Update.
	[H Rc y] = fn('hry', it);
	[xf Pfc] = kf_qrsc_update(H, Rc, y, xp, Ppc);
      end
      
     case {'firsturn' 'youturn'}
      it = capo;
      dbg('      reverse %d\n', it);

      % Predict.
      if (it > 1)
        [F Q] = fn('fq', it);
        [xp Ppc] = kf_qrsc_predict(F, Q, xf, Pfc);
      end
      % Update.
      [H Rc y] = fn('hry', it);
      z = y - H*xp;
      Sc = H*Ppc';
      Sc = chol(Rc'*Rc + Sc*Sc');
      % Contribution to loglik.
      loglik = loglik + fn('ll', it, Sc, z);
      % Optional user output function.
      if (hofn) ofn(it, xp, Ppc, xf, Pfc, z, Sc); end

      % Reverse.
      Pp = Ppc'*Ppc;
      Si = linsolve(full(Sc), eye(size(Sc, 1)), opts); Si = Si*Si'; % inv(S)
      [p_S p_z] = fn('p', it, Sc, Si, z);
      if (it == Nt) Fp1 = sparse(Ns, Ns); else Fp1 = fn('f', it+1); end
      [lambda mu eta beta gamma] = kf_grad( ...
        Fp1, H, Sc, Si, p_S, p_z, z, Pp, mu, beta);
      git = fn('g', it, lambda, mu, eta, beta, gamma);
      if (isempty(g)) g = git; else g = g + git; end
      
     case 'terminate'
      break;
      
     case 'error'
      error('REVOLVE reports error.');
      
    end
  end
end
