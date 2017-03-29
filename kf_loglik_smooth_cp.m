function loglik = kf_loglik_smooth_cp(rcd,fn,ofn,Ns,Nt,Nbytes)
% loglik = kf_loglik_smooth_cp(rcd,fn,ofn,Ns,Nt,Nbytes)
%   Compute the RTS smoother.
%
%   This routine uses a mex interface to Andreas Griewank's checkpointing
% software REVOLVE to limit storage.
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
%     [F Q]     = fn('fq' ,tidx)
%       Return the state transition matrix F and possibly the process noise Q
%       at time step 1 <= tidx <= Nt.
%     [H Rc y]  = fn('hry',tidx)
%       Return the observation matrix H, the Cholesky factorization of the
%       observation noise R (chol(R)), and the observations y at time step
%       tidx.
%     p         = fn('ll', tidx,Sc,z)
%       Return the contribution of time step tidx to the log likelihood given
%       Sc = chol(R + H Pp H') and z = y - H xp.
% These calls are a subset of those used by kf_loglik_grad.
%   ofn is a function handle that implements the following call:
%     ofn(it,xp,Ppc,xf,Pfc,z,Sc,xs,Psc)
% It is called each time a smoothing step is taken. The user can save desired
% output for step 'it'.
%
% See ex.m for an example of usage.
  
  %dbg = @(varargin)fprintf(1,varargin{:});
  dbg = @(varargin)1;
  % Init stack.
  Nsnaps = min(Nt,floor(Nbytes/((Ns+1)*Ns*8)));
  stack = zeros(Ns,Ns+1,Nsnaps);
  % Init log likelihood.
  loglik = 0;
  % Init REVOLVE parameters.
  capo = 1;
  check = -1;
  fine = Nt + capo;
  info = 0;
  done = false;
  while (~done)
    oldcapo = capo;
    [action check capo fine info] = revolve(check,capo,fine,Nsnaps,info);
    dbg('%10s %5d %5d\n',action,capo,check);
    switch (action)
     
     case 'takeshot'
      prev_check = check;
      if (capo ~= 1)
        rcd = kf_rcd('push',rcd,Pfc,xf);
      else
	[xp Ppc] = fn('i');
        rcd = kf_rcd('push',rcd,Ppc,xp);
      end
     
     case 'restore'
      if (check < prev_check)
        rcd = kf_rcd('discard',rcd);
      end
      prev_check = check;
      if (capo ~= 1)
        [rcd Pfc xf] = kf_rcd('peak',rcd);
      else
        [rcd Ppc xp] = kf_rcd('peak',rcd);
      end
      
     case 'advance'
      % Forward.
      dbg('      forward %d:%d\n',oldcapo,capo-1);
      for (it = oldcapo:capo-1)
	% Predict.
	if (it > 1)
	  [F Q] = fn('fq',it);
	  [xp Ppc] = kf_qrsc_predict(F,Q,xf,Pfc);
	end
	% Update.
	[H Rc y] = fn('hry',it);
	[xf Pfc] = kf_qrsc_update(H,Rc,y,xp,Ppc);
      end

     case {'firsturn' 'youturn'}
      it = capo;
      dbg('      reverse %d\n',it);

      % Predict.
      if (it > 1)
        [F Q] = fn('fq',it);
	[xp Ppc] = kf_qrsc_predict(F,Q,xf,Pfc);
      end
      % Update.
      [H Rc y] = fn('hry',it);
      [xf Pfc z Sc] = kf_qrsc_update(H,Rc,y,xp,Ppc);
      % Contribution to loglik.
      loglik = loglik + fn('ll',it,Sc,z);

      % Reverse.
      if (it < Nt)
        [F Q] = fn('fq',it+1);
	[xp Ppc] = kf_qrsc_predict(F,Q,xf,Pfc);
	try
	  [xs Psc] = kf_chol_smooth(F,xf,xs,Ppc,Pfc,Psc);
	catch
	  % Lost definiteness using chol only, so use QR-based square root
	  % formulation.
          warning('Using kf_qrsc_smooth, as kf_chol_smooth failed.\n');
          [xs Psc] = kf_qrsc_smooth(F,chol(Q),xf,xs,Ppc,Pfc,Psc);
	end
      else
	xs = xf;
	Psc = Pfc;
      end
      ofn(it,xp,Ppc,xf,Pfc,z,Sc,xs,Psc);
      
     case 'terminate'
      break;
      
     case 'error'
      error('REVOLVE reports error.');
      
    end
  end
end