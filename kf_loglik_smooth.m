function loglik = kf_loglik_smooth(rcd,fn,ofn,Ns,Nt,Nbytes)
% loglik = kf_loglik_smooth(rcd,fn,ofn,Ns,Nt,Nbytes)
%   Compute the RTS smoother.
%
% This version does not use checkpointing. See kf_loglik_smooth_cp for a
% description of the inputs to this routine.

% AMB ambrad@cs.stanford.edu
% CDFM, Dept of Geophysics, Stanford

  % Forward.
  loglik = 0;
  for (it = 1:Nt)
    % Predict.
    if (it > 1)
      [F Q] = fn('fq',it);
      [xp Ppc] = kf_qrsc_predict(F,Q,xf,Pfc);
    else
      [xp Ppc] = fn('i');
    end
    % Update.
    [H Rc y] = fn('hry',it);
    [xf Pfc z Sc] = kf_qrsc_update(H,Rc,y,xp,Ppc);
    % Contribution to loglik.
    loglik = loglik + fn('ll',it,Sc,z);
    % Record (for kfgs internal use).
    if (it < Nt)
      rcd = kf_rcd('push',rcd,Pfc,xf);
    end
  end

  % Reverse.
  for (it = Nt:-1:1)
    if (it < Nt)
      [rcd Pfc xf] = kf_rcd('pop',rcd);
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
  end
end
