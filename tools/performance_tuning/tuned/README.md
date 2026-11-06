# Tuned built-in scenes

Every built-in viewer scene, tuned by `tools/performance_tuning/tune_all.py`
on an RTX 4080 (16 GiB). 

- **baseline**: the scene's own settings -- the per-world MJWarp capacities the viewer sized
  for one world, the collision defaults (fractions 1.0, tile size 8), the solver defaults --
  at the family's batch size, or the largest that fit.
- **tuned**: the file's settings at the batch size the sweep found fastest. *Changed* lists
  what besides the capacities the tuning changed, each kept only past the artifact guard
  (`common.no_worse`); *capacities* the main ones (Lodos's tile and stage-2 fractions,
  MJWarp's contacts and rows).
- **check**: the final artifact check of the tuned settings against the baseline's.

World-steps per second (k = 10^3, M = 10^6).

| scene | B | baseline | B | tuned | ratio | changed | capacities | check |
|---|---|---|---|---|---|---|---|---|
| `incline_sdf` | 2048 | 86.4k | 8192 | 1.81M | 20.92 | prox_iters_per_qp=3, contact_reduction | tile 0.612, c2 0.36, nacon 708432, njmax 365 | yes |
| `incline_polytope` | 2048 | 458k | 16384 | 4.5M | 9.82 | prox_iters_per_qp=5 | nacon 98304, njmax 18 | yes |
| `painleve_box_sdf` | 2048 | 99.4k | 16384 | 1.74M | 17.56 | prox_iters_per_qp=3, contact_reduction | tile 0.374, c2 0.68, nacon 570573, njmax 117 | yes |
| `painleve_box_polytope` | 2048 | 474k | 16384 | 5.89M | 12.42 | prox_iters_per_qp=5 | nacon 98304, njmax 18 | yes |
| `backspin_ball_sdf` | 2048 | 109k | 16384 | 1.57M | 14.50 | tile sizes per pair | tile 0.326, c2 0.0223, nacon 78174, njmax 729 | yes |
| `backspin_ball_polytope` | 8192 | 499k | 32768 | 4.63M | 9.29 | -- | nacon 49152 | yes |
| `masonry_arch_sdf` | 128 | 9.01k | 2048 | 27.5k | 3.06 | compacted tile filter | tile 0.0442, c2 0.648, nacon 340203, njmax 3213 | yes |
| `masonry_arch_polytope` | 1024 | 37.4k | 4096 | 48.4k | 1.30 | -- | nacon 543488, njmax 419 | yes |
| `house_of_cards_sdf` | 128 | 2.31k | 2048 | 4.66k | 2.02 | -- | tile 0.0949, c2 0.0246, nacon 3136565, njmax 5909 | yes |
| `house_of_cards_polytope` | 1024 | 13.8k | 4096 | 16.3k | 1.18 | -- | nacon 952355, njmax 743 | yes |
| `cube_5_sdf` | 2048 | 18.7k | 2048 | 110k | 5.89 | prox_iters_per_qp=5, contact_reduction | tile 0.0959, c2 0.279, nacon 426362, njmax 1683 | yes |
| `cube_5_polytope` | 2048 | 320k | 4096 | 614k | 1.92 | prox_iters_per_qp=5 | nacon 75393, njmax 99 | yes |
| `cube_walled_5_sdf` | 2048 | 23.5k | 2048 | 77k | 3.28 | contact_reduction, tile sizes per pair | tile 0.0878, c2 0.157, nacon 261828, njmax 1553 | yes |
| `cube_walled_5_polytope` | 2048 | 296k | 4096 | 561k | 1.89 | prox_iters_per_qp=5 | nacon 77987, njmax 104 | yes |
| `cube_20_sdf` | 512 | 6.13k | 4096 | 32.3k | 5.27 | prox_iters_per_qp=5, contact_reduction, tile sizes per pair | tile 0.0798, c2 0.0921, nacon 2315442, njmax 3969 | yes |
| `cube_20_polytope` | 2048 | 93k | 8192 | 251k | 2.70 | prox_iters_per_qp=3 | nacon 670815, njmax 378 | yes |
| `cube_walled_20_sdf` | 256 | 5.21k | 4096 | 17.6k | 3.37 | contact_reduction, tile sizes per pair | tile 0.0315, c2 0.19, nacon 2303580, njmax 3938 | yes |
| `cube_walled_20_polytope` | 2048 | 86.7k | 4096 | 217k | 2.50 | prox_iters_per_qp=3 | nacon 333546, njmax 401 | yes |
| `arm_5_sdf` | 2048 | 80.6k | 4096 | 309k | 3.83 | -- | tile 0.134, c2 0.0259, nacon 51098, njmax 77 | yes |
| `arm_5_polytope` | 1024 | 119k | 4096 | 355k | 2.97 | prox_iters_per_qp=5 | nacon 414797, njmax 198 | yes |
| `arm_walled_5_sdf` | 2048 | 70.1k | 4096 | 203k | 2.90 | tile sizes per pair | tile 0.183, c2 0.0185, nacon 70946, njmax 90 | yes |
| `arm_walled_5_polytope` | 1024 | 87.1k | 2048 | 130k | 1.49 | -- | nacon 309851, njmax 189 | yes |
| `arm_20_sdf` | 256 | 12.4k | 4096 | 118k | 9.47 | prox_iters_per_qp=5, tile sizes per pair | tile 0.0995, c2 0.011, nacon 247853, njmax 297 | yes |
| `arm_20_polytope` | 256 | 15.2k | 4096 | 56k | 3.68 | -- | nacon 1964769, njmax 432 | yes |
| `arm_walled_20_sdf` | 256 | 11.9k | 4096 | 62.8k | 5.27 | tile sizes per pair | tile 0.0663, c2 0.0178, nacon 275061, njmax 302 | yes |
| `arm_walled_20_polytope` | 256 | 14.1k | 4096 | 46.3k | 3.28 | -- | nacon 2384774, njmax 459 | yes |
| `sharpa_5_sdf` | 128 | 5.93k | 4096 | 33.5k | 5.66 | tile sizes per pair | tile 0.015, c2 0.0128, nacon 300891, njmax 905 | yes |
| `sharpa_5_polytope` | 512 | 22.2k | 8192 | 90.9k | 4.09 | prox_iters_per_qp=5 | nacon 3271269, njmax 1023 | yes |
| `sharpa_walled_5_sdf` | 128 | 5.8k | 4096 | 31.7k | 5.46 | tile sizes per pair | tile 0.0144, c2 0.0136, nacon 315569, njmax 905 | yes |
| `sharpa_walled_5_polytope` | 512 | 22k | 4096 | 100k | 4.56 | prox_iters_per_qp=3 | nacon 1677125, njmax 1235 | yes |
| `sharpa_20_polytope` | 128 | 442 | 1024 | 1.76k | 3.97 | prox_iters_per_qp=5 | nacon 1631294, njmax 2636 | yes |
| `sharpa_walled_20_polytope` | 128 | 439 | 1024 | 1.75k | 3.98 | prox_iters_per_qp=5 | nacon 1630178, njmax 2649 | yes |
| `shadow_5_sdf` | 128 | 5.57k | 4096 | 27.9k | 5.01 | tile sizes per pair | tile 0.0258, c2 0.0148, nacon 313527, njmax 1446 | yes |
| `shadow_5_polytope` | 512 | 21.8k | 4096 | 106k | 4.85 | prox_iters_per_qp=3 | nacon 2174867, njmax 1016 | yes |
| `shadow_walled_5_sdf` | 128 | 5.37k | 4096 | 27.4k | 5.11 | tile sizes per pair | tile 0.0245, c2 0.0141, nacon 305694, njmax 1422 | yes |
| `shadow_walled_5_polytope` | 512 | 21.4k | 4096 | 102k | 4.77 | prox_iters_per_qp=3 | nacon 2188595, njmax 1095 | yes |
| `shadow_20_polytope` | 128 | 420 | 1024 | 2.64k | 6.29 | prox_iters_per_qp=3 | nacon 2148596, njmax 2577 | yes |
| `shadow_walled_20_polytope` | 128 | 419 | 1024 | 2.72k | 6.50 | prox_iters_per_qp=3 | nacon 2117346, njmax 2687 | yes |
| `sharpa_armadillo_sdf` | 1024 | 63.3k | 4096 | 173k | 2.74 | -- | tile 0.0456, c2 0.0372, nacon 68607, njmax 131 | yes |
| `sharpa_armadillo_polytope` | 1024 | 170k | 4096 | 491k | 2.88 | prox_iters_per_qp=3 | nacon 377718, njmax 194 | yes |
| `shadow_grasp_sdf` | 1024 | 164k | 8192 | 1.43M | 8.67 | prox_iters_per_qp=3 | tile 0.00894, c2 0.00616, nacon 8192, njmax 75 | yes |
| `shadow_grasp_polytope` | 1024 | 200k | 8192 | 1.55M | 7.73 | prox_iters_per_qp=3 | nacon 87770, njmax 105 | yes |
| `nist_threading_m8` | 1024 | 396k | 4096 | 859k | 2.17 | -- | tile 0.0874, c2 0.155, nacon 143807, njmax 635 | yes |
| `nist_threading_m8_easy` | 1024 | 335k | 4096 | 613k | 1.83 | -- | tile 0.113, c2 0.314, nacon 206598, njmax 734 | yes |
| `nist_nutbolt_m8` | 1024 | 54.8k | 8192 | 850k | 15.50 | tile sizes per pair | tile 0.44, c2 0.308, nacon 131408, njmax 243 | yes |
| `nist_nutbolt_m8_s0.25` | 1024 | 52.7k | 8192 | 655k | 12.42 | prox_iters_per_qp=3, tile sizes per pair | tile 0.308, c2 0.48, nacon 215442, njmax 324 | yes |
| `nist_nutbolt_m8_s0.125` | 512 | 39.4k | 4096 | 251k | 6.38 | prox_iters_per_qp=5, tile sizes per pair | tile 0.262, c2 0.552, nacon 115340, njmax 401 | yes |
| `nist_nutbolt_m12` | 1024 | 55k | 8192 | 1.2M | 21.90 | prox_iters_per_qp=3, tile sizes per pair | tile 0.419, c2 0.298, nacon 174588, njmax 239 | yes |
| `nist_nutbolt_m12_s0.35` | 1024 | 52k | 8192 | 657k | 12.64 | prox_iters_per_qp=3, tile sizes per pair | tile 0.296, c2 0.469, nacon 163664, njmax 275 | yes |
| `nist_nutbolt_m12_s0.175` | 512 | 38.3k | 4096 | 255k | 6.65 | prox_iters_per_qp=5, tile sizes per pair | tile 0.247, c2 0.528, nacon 93741, njmax 351 | yes |
| `nist_nutbolt_m16` | 1024 | 56.8k | 8192 | 1.42M | 25.06 | prox_iters_per_qp=3 | tile 0.327, c2 0.263, nacon 92949, njmax 212 | yes |
| `nist_nutbolt_m16_s0.4` | 1024 | 52.4k | 4096 | 595k | 11.35 | prox_iters_per_qp=3, tile sizes per pair | tile 0.416, c2 0.183, nacon 56043, njmax 198 | yes |
| `nist_nutbolt_m16_s0.2` | 512 | 39.4k | 4096 | 310k | 7.86 | prox_iters_per_qp=5, tile sizes per pair | tile 0.206, c2 0.286, nacon 66990, njmax 243 | yes |
| `nist_driven_m8` | 1024 | 58.1k | 8192 | 1.45M | 25.03 | prox_iters_per_qp=3, tile sizes per pair | tile 0.355, c2 0.291, nacon 212649, njmax 131 | yes |
| `nist_driven_m12` | 1024 | 56.4k | 8192 | 1.58M | 28.09 | prox_iters_per_qp=3, tile sizes per pair | tile 0.338, c2 0.278, nacon 157404, njmax 153 | yes |
| `nist_driven_m16` | 1024 | 60.5k | 8192 | 1.77M | 29.27 | prox_iters_per_qp=3 | tile 0.282, c2 0.229, nacon 112047, njmax 81 | yes |
| `nist_board` | 512 | 23.9k | 2048 | 244k | 10.18 | prox_iters_per_qp=3 | tile 0.527, c2 0.262, nacon 142203, njmax 455 | yes |
| `arm_box_6` | 1024 | 123k | 4096 | 244k | 1.99 | -- | tile 0.14, c2 0.0253, nacon 272147, njmax 113 | yes |
| `arm_box_20` | 128 | 12.1k | 4096 | 59.1k | 4.89 | -- | tile 0.0919, c2 0.0271, nacon 1074806, njmax 356 | yes |
| `nutbolt_05` | 1024 | 44.2k | 2048 | 177k | 4.01 | -- | tile 0.206, c2 0.402, nacon 144756, njmax 392 | yes |
| `nutbolt_05_trilinear` | 1024 | 46.3k | 2048 | 145k | 3.14 | -- | tile 0.127, c2 0.354, nacon 217358, njmax 356 | yes |
| `nutbolt_1` | 1024 | 47.8k | 2048 | 199k | 4.16 | -- | tile 0.223, c2 0.474, nacon 148233, njmax 365 | yes |
| `nutbolt_1_trilinear` | 1024 | 47.1k | 2048 | 182k | 3.86 | -- | tile 0.192, c2 0.462, nacon 169508, njmax 360 | yes |
| `sharpa_2_sdf` | 512 | 27.6k | 2048 | 200k | 7.23 | prox_iters_per_qp=5, tile sizes per pair | tile 0.0234, c2 0.0144, nacon 55896, njmax 678 | yes |
| `sharpa_2_polytope` | 1024 | 114k | 4096 | 453k | 3.98 | prox_iters_per_qp=3 | nacon 558597, njmax 587 | yes |
| `sharpa_walled_2_sdf` | 512 | 27.4k | 2048 | 130k | 4.72 | tile sizes per pair | tile 0.0178, c2 0.0148, nacon 51971, njmax 530 | yes |
| `sharpa_walled_2_polytope` | 1024 | 111k | 4096 | 448k | 4.04 | prox_iters_per_qp=3 | nacon 540845, njmax 587 | yes |
