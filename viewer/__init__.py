"""
The viewer: an interactive session -- simulate, edit the state, switch sims, scrub takes (`app`,
`session`) -- with the scenes it loads (`scenes`, `scenes.catalog`) and their assets (`viewer/assets/`).
"""

# The drawing (`draw`) and the recording it scrubs (`trajectory`) are host-side numpy; the
# session drives a `lodos.sim.Sim`.
from viewer import trajectory
