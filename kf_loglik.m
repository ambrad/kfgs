function loglik = kf_loglik(fn,Ns,Nt,ofn)
% loglik = kf_loglik(fn,Ns,Nt,ofn)
%   Kalman filter.
%
% See kf_loglik_smooth_cp for a description of the inputs.
  
  hofn = nargin > 3;
  
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
    if (hofn) ofn(it,xp,Ppc,xf,Pfc,z,Sc); end
  end
end
