function R = kf_grad_Calc_invC(R)
% R = kf_grad_Calc_invC(R)
% For C = R' R, return inv(C).
  R = inv(R);
  R = R*R';
end
