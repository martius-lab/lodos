"""
MJWarp's constraint solver, adapted from MuJoCo Warp 3.14.0's mujoco_warp/_src (Apache-2.0), with Lodos's proximal
solve and contact groups added; each module's header names its source files. Imports no submodule.

  constraint       the constraint rows: contacts (and contact groups), equality, friction
                   loss and limits, MuJoCo's weights and references (`make_constraint`)
  nlsp             nonlinear least squares with proximal minimization: MJWarp's CG and
                   Newton solve, with the prox solver's proximal loop around it under
                   `opt.prox_enable` (`solve`); MuJoCo Warp's solver.py with the
                   proximal solve fused in
  block_cholesky   the blocked Cholesky factor-and-solve the Newton solve and
                   `lodos.cd.rowspace` use
"""
