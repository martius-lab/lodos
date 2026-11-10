"""
Scene builders.

  drop              cubes or convex-decomposed armadillos dropped into a walled box, all MJCF geoms (MJWarp's pipeline)
  armadillo         five armadillos dropped onto a plane, on baked surfaces only (Lodos's pipeline)
  representations   one drop of armadillos or cubes in either collision representation, with or without walls
  mixed             both at once: armadillo surfaces supplemented by their convex pieces, among MJCF cubes and walls
  fbf               the FBF paper's exactness scenes: incline, backspin, Painleve, cards, arch; either representation
  toa               thin features under time of arrival: cubes at a card, past its edge, in rings, on hammers; a bullet
  sharpa            SHARPA hands dropped on a floor or into a walled box
  sharpa_armadillo  the SHARPA hand holding a 70 mm armadillo, palm up
  shadow            Shadow hands dropped on a floor or into a walled box
  shadow_kong       the Shadow hand lifting a Kong toy in Dexonomy's four-finger prismatic grasp
  nutbolt           Newton's nut-bolt example: IsaacGymEnvs' M20 nut screwing itself down its bolt
  nist              Isaac Lab's NIST parts: the M8, M12, M16 nut-bolt pairs, the board, the driven nut, the threading
  franka            the Franka Panda of Isaac Lab's NIST demo, for `nist`

`catalog` makes the viewer's scenes of them, or of an MJCF file; `assets` finds what they load.
"""
