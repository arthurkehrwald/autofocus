# Autofocus

Finds the sharpest image in a list of images to help automatically focus camera
lenses. At least one printed fiducial marker (marker.pdf) must be visible in
each image without any obstruction at the distance that should be in focus.
The largest marker is recognized using the Aruco markers located at each corner
and cropped out. The sharpness is rated using the Laplacian variance of that
cropped image.
