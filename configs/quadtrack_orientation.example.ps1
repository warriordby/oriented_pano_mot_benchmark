$QuadTrackRoot = "D:\googledownload\omnitrack\QuadTrack_test\OmniTrack_Omnidet_test"
$OutRoot = "D:\googledownload\trackers\oriented_pano_mot_benchmark\outputs\quadtrack_orientation_benchmark"

py -B tools\convert_quadtrack_to_orientation_benchmark.py `
  --quadtrack-root $QuadTrackRoot `
  --out-root $OutRoot `
  --image-width 2048 `
  --image-height 480 `
  --vertical-fov-deg 120 `
  --variants prior_a2b,polar_up,target_north_80 `
  --edge-samples 32 `
  --mot-frame-to-image-offset -1 `
  --frame-name-width 6 `
  --frame-image-ext .jpg `
  --input-kind detections
