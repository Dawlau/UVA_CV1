USAGE

The images provided in this dataset are free to use so long as proper
credit is attributed to the original authors. In particular, any
publications utilizing this dataset should cite the following paper:

N. Alldrin, T. Zickler, and D. Kriegman, "Photometric Stereo With
Non-Parametric and Spatially-Varying Reflectance", 2008 Conf. on
Comp. Vision and Pattern Recognition (CVPR), Anchorage, AK, June 2008.


FILE NAMES / CONVENTIONS

* I_????.exr
HDR images in exr format, numbered 0 to N

* mask.png
A binary image (values either 0 or 1) with 1 indicating foreground, 0
background.

* light_directions.txt
Each row consists of x,y,z coordinates of a unit vector specifying the
direction of the corresponding light source. The x and y axes span the
image plane (x points right, y points up) and the z-axis points into
the camera along the optical axis.

* light_intensities.txt
Each row contains the intensity of the corresponding light
source. While we did not measure the spectral composition of the
lighting (a 100W clear incandescent bulb), we record each intensity
three times (corresponding to r,g,b color channels) so that our file
format is consistent with that in the USC light stage data gallery.

* light_directions_refined.txt / light_intensities_refined.txt
Due to measurement errors, the light source positions and intensities
are slightly incorrect. While fitting the shape / reflectance of the
test object we also iteratively refined the light source positions and
intensities. The refined light source positions are believed to be
much more accurate than the originally measured positions.


DETAILS OF IMAGE ACQUISITION

Images were acquired in a dark room using an EOS-1Ds camera with a
fixed zoom lens. The camera was placed roughly 1.5m from the test
object (~10cm in diameter). All images were acquired with the camera
and test object in the same position. Illumination consisted of a
single incandescent light bulb placed roughly 1.5m from the test
object (whose position was varied in each image). Light source
directions were measured with a mirrored sphere placed in the scene
and light source intensity was measured with a diffuse sphere placed
in the scene.

Each image in the dataset is actually the result of multiple RAW
images captured by the camera. To prevent clipping of bright and dark
regions of the scene, we combined multiple low dynamic range (LDR)
images taken at different shutter speeds into a single high dynamic
range (HDR) image. To eliminate ambient illumination (which is present
even in a dark room to some degree), we acquired "ambient images" by
occluding the light source relative to the test object (i.e., we
blocked the light source so that the test object was in shadow, so
that only ambient light illuminated the object). The images present in
the dataset are the difference between normal images and their
corresponding ambient images.

