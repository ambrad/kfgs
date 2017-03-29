function R = kf_grad_Calc_ztinvCz_C(z,R)
% R = kf_grad_Calc_ztinvCz_C(z,R)
% For C = R' R, return
%     partial_C (z' inv(C) z).
% The formula follows from applying the product rule to
%     C inv(C) = I.
  q = R \ (z' / R)';
  R = -q*q';
end
