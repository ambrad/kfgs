function varargout = ex (varargin)
% Example usage of KFGS.
%
% ex('ex_grad', Ns, No, Nt, Nstack)
%   Compute the log-likelihood and gradient for a random problem having state
% vector size Ns, observation vector size No, time steps Nt, and at most
% Nstack saved intermediate steps.
  [varargout{1:nargout}] = feval(varargin{:});
end

function ex_grad (Ns, No, Nt, Nstack)
  % Set up a random problem. p holds F, H, Q, R, etc.
  p = RandProb(Ns, No, Nt);
  % 1 or 2. Just test different types of parameterizations.
  p.grad_test = 2;
  % Run the basic Kalman filter for timing comparison.
  t = tic();
  ll1 = KalmanFilter(p.F, p.Q, p.H, p.Rc, p.x0, p.Pp0c, p.Y);
  etkf = toc(t);
  % Now run kf_loglik_grad to get both the log likelihood and its gradient
  % for the nominal set of parameters.
  t = tic();
  r = kf_rcd('init', 'm');
  [ll2 ga] = kf_loglik_grad_cp(...
      r, @(varargin)kl_fn(p, varargin{:}), Ns, Nt, ...
      Nstack*(Ns+1)*Ns*8);
  eta = toc(t);
  % Compare.
  fprintf(1, 'et = %1.2fs  time compared with KF = %1.2fx\n', eta, eta/etkf);
end

function ex_smooth (Ns, No, Nt, Nstack)
  % Set up a random problem.
  p = RandProb(Ns, No, Nt);
  % Run the basic Kalman filter for timing comparison.
  t = tic();
  ll1 = KalmanFilter(p.F, p.Q, p.H, p.Rc, p.x0, p.Pp0c, p.Y);
  etkf = toc(t);
  t = tic();
  % Run the KF/smoother.
  clear global gs;
  r = kf_rcd('init', 'm');
  ll2 = kf_loglik_smooth_cp(...
      r, @(varargin)kl_fn(p, varargin{:}), ...
      @kl_ofn, Ns, Nt, Nstack*(Ns+1)*Ns*8);
  eta = toc(t);
  % Compare.
  fprintf(1, 'et = %1.2fs  time compared with KF = %1.2fx\n', eta, eta/etkf);
  % In practice, we would access gs data here if desired.
  % ...
  % Then clear it.
  clear global gs;
end
  
function p = RandProb (m, n, nt)
  p.F = randn(m);
  p.H = randn(n, m);
  p.Pp0 = randn(m); p.Pp0 = p.Pp0'*p.Pp0; p.Pp0c = chol(p.Pp0);
  p.x0 = randn(m, 1);
  p.Y = randn(n, nt);
  p.Q = randn(m); p.Q = p.Q'*p.Q;
  p.R = randn(n); p.R = p.R'*p.R; p.Rc = chol(p.R);
end
  
function loglik = KalmanFilter (F, Q, H, Rc, x0, Pp0c, Y)
  nt = size(Y, 2);
  ns = size(F, 1);
  loglik = 0;
  for (it = 1:nt)
    if (it > 1)
      [xp Ppc] = kf_qrsc_predict(F, Q, xf, Pfc);
    else
      xp = x0;
      Ppc = Pp0c;
    end
    [xf Pfc z Sc] = kf_qrsc_update(H, Rc, Y(:, it), xp, Ppc);
    q = z' / Sc;
    loglik = loglik - sum(log(diag(Sc))) - 0.5*q*q';
  end
end

function varargout = kl_fn (p, key, tidx, varargin)
  switch (key)
    case 'i'
      % Initial conditions.
      varargout(1:2) = {p.x0 p.Pp0c};
      
    case 'f'
      % State-space transition matrix.
      varargout{1} = p.F;
      
    case 'fq'
      % State-space transition matrix and process covariance.
      varargout(1:2) = {p.F p.Q};
      
    case 'h'
      % Observation matrix.
      varargout{1} = p.H;
      
    case 'hry'
      % Observation matrix, chol(observation covariance), observations at
      % time index tidx.
      varargout(1:3) = {p.H p.Rc p.Y(:, tidx)};
      
    case 'll'
      % Contribution to log-likelihood at time index tidx.
      [Sc z] = deal(varargin{1:2});
      q = z' / Sc;
      varargout{1} = -sum(log(diag(Sc))) - 0.5*q*q';
      
    case 'p'
      % Partial derivatives of the log-likelihood term at time index tidx
      % with respect to S = R + H Pp H' and z = y - H xp.
      [Sc Si z] = deal(varargin{1:3});
      p_S = -0.5*(kf_grad_Calc_LogDetC_C(Si) + kf_grad_Calc_ztinvCz_C(z, Sc));
      p_z = -Sc \ (z' / Sc)';
      varargout = {p_S p_z};
      
    case 'g'
      % Contribution of the term for time index tidx to the gradient of the
      % log-likelihood function.
      %   If tidx == 1, there was no prediction and so no Q. However, there was
      % a filter ('update').
      [lambda mu] = deal(varargin{1:2});
      switch (p.grad_test)
        case 1
          % Gradient wrt to diagonal elements of R and Q.
          [m n] = size(p.H);
          if (tidx > 1) mu1 = diag(reshape(mu, n, n));
          else          mu1 = zeros(n, 1); end
          varargout{1} = -[mu1; diag(reshape(lambda, m, m))].';
        case 2
          % Gradient wrt a scalar multiple of R and Q.
          if (tidx > 1) mu1 = mu(:)'*p.Q(:);
          else          mu1 = 0; end
          varargout{1} = -[mu1, lambda(:)'*p.R(:)];
      end
  end
end

function kl_ofn (it, xp, Ppc, xf, Pfc, z, Sc, xs, Psc)
% I do the simplest thing possible: just save certain parts of the data to a
% global struct. More complicated ideas:
%   - save smoothed state always, but the full cov matrix only every Kth time
%     step
%   - write data to disk; this is necessary if the total amount of data to
%     save is large.
  global gs;
  gs.xss(:, it) = xs;
  gs.vars(:, it) = diag(Psc'*Psc);
end
