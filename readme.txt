KFGS: Software to compute the
      (K)alman (F)ilter, Likelihood (G)radient, and (S)moother
Version 0.4
Andrew M. Bradley
ambrad@cs.stanford.edu
CDFM, Geophysics, Stanford

This package provides:

1. Pure Matlab routines to compute stably
   a. the Kalman filter (KF),
   b. the gradient of the log likelihood of the data in a time proportional to
      the time to compute the KF, with a constant of about 1-2.
   c. the same as (b), but with a constant of about 1-10 in practice and amount
      of storage that is very small compared with the number of time steps.
   c. the RTS smoother, with versions like (b) or (c).
2. A mex wrapper to Andreas Griewank's checkpointing software REVOLVE.

KFGS is licensed as follows:
Open Source Initiative OSI - Eclipse Public License 1.0
http://www.opensource.org/licenses/eclipse-1.0


Matlab Installation
-------------------
You need an ANSI C compiler if you want to use the checkpointing version of the
gradient calculation. The most basic mex setup possible should work. At the
command line, type "make":
    >> make


Usage
-----
See ex.m for an example of usage. The primary public functions are
    kf_loglik:           Kalman filter.
    kf_loglik_grad:      Likelihood gradient using the adjoint method.
    kf_loglik_grad_cp:   Same, but with checkpointing.
    kf_loglik_smooth:    RTS smoother.
    kf_loglik_smooth_cp: Same, but with checkpointing.
    kf_rcd('init'):      Initialize memory or disk recording for the gradient
                         and smoother reoutines.

The checkpointing version of a function uses the mex file revolve.c. My
experience is that in general the non-checkpointing version with disk I/O is
faster than the checkpointing version using main memory but number of
checkpoints < number of time steps. Hence I think the only reasons to use the
checkpointing version are (1) the amount of disk storage necessary is greater
than you have or (2) I/O is particularly slow relative to FLOPS and memory
access.
  Example: Nstate = 600, Nobs = 200, Nt = 3650.
1. kf_loglik_smooth_cp with a buffer equivalent to saving 100 states took 1613s.
2. kf_loglik_smooth saving to disk a total of ~9.8G took 1162s.

If you use this software in research that you publish, I would appreciate your
citing this software package as follows:

    A. M. Bradley, "KFGS: Software to Compute the Kalman Filter Likelihood
    Gradient and Smoother", 2012.

Email me (ambrad@cs.stanford.edu) if you want to receive an email when I release
a new version.


Important version changes
-------------------------


Release log
-----------
0.0.  AMB. 28 May 2012. First release.
0.1.  AMB.
      - _cp and no _cp versions of gradient and smoother.
      - kf_rcd for storage management.
0.2.  AMB.
      - using linsolve rather than \ in a few places.
0.3.  AMB.
0.4.  AMB. More docs than 3, some cosmetics.

To do
-----
- A subset of the operations in kf_grad.m could take advantage of symmetry for a
  2x reduction in work and memory. Not a big deal since these are O(N^2), not
  O(N^3) operations.
- I could generalize the gradient computation so multiple, rather than just one,
  quantity's gradient can be found. This would let us implement the Segall &
  Matthews 97 trick of removing an overall scaling from the problem. That trick
  requires accumulating two quantities and taking the log of one before summing
  them. Only one quantity is accumulated for the standard likelihood function.
- Hessian?
