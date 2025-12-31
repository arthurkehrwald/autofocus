"""
Finds the sharpest image in a list of images to help automatically focus camera
lenses. At least one printed fiducial marker (marker.pdf) must be visible in
each image without any obstruction at the distance that should be in focus.
The largest marker is recognized using the Aruco markers located at each corner
and cropped out. The sharpness is rated using the Laplacian variance of that
cropped image.
"""

from pathlib import Path
import typing
import cv2 as cv


class Point(typing.NamedTuple):
    x: int
    y: int


class MarkerDetection(typing.NamedTuple):
    marker_id: int
    corners: typing.List[Point]


def collect_image_paths(dirpath: Path) -> typing.List[Path]:
    exts = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff", "*.bmp")
    files = []
    for e in exts:
        files.extend(sorted(dirpath.glob(e)))
    return files


def collect_images(img_paths: typing.List[Path]) -> typing.List[cv.typing.MatLike]:
    return [cv.imread(str(path)) for path in img_paths]


def find_aruco_markers(img: cv.typing.MatLike) -> typing.List[MarkerDetection]:
    dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_250)
    parameters = cv.aruco.DetectorParameters()
    detector = cv.aruco.ArucoDetector(dictionary, parameters)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    corners_list, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return []
    return [
        MarkerDetection(
            marker_id=int(ids[i][0]),
            corners=[Point(int(c[0]), int(c[1])) for c in corners_list[i][0]],
        )
        for i in range(len(ids))
    ]


def find_test_markers(img: cv.typing.MatLike) -> typing.List[MarkerDetection]:
    corner_markers = find_aruco_markers(img)
    test_markers = {}
    for marker in corner_markers:
        if marker.marker_id not in test_markers:
            test_markers[marker.marker_id] = []
        test_markers[marker.marker_id].append(marker.corners[0])

    return [
        MarkerDetection(marker_id=mid, corners=corners)
        for mid, corners in test_markers.items()
        if len(corners) == 4
    ]


def calc_area(marker: MarkerDetection) -> float:
    # Calculate area using the shoelace formula for a polygon
    corners = marker.corners
    n = len(corners)
    area = 0.0
    for i in range(n):
        x1, y1 = corners[i]
        x2, y2 = corners[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def find_best_test_marker(img: cv.typing.MatLike) -> MarkerDetection:
    markers = find_test_markers(img)
    if not markers:
        return None
    best_marker = max(markers, key=calc_area)
    return best_marker


def crop_out_marker(
    img: cv.typing.MatLike, marker: MarkerDetection
) -> cv.typing.MatLike:
    x_coords = [point.x for point in marker.corners]
    y_coords = [point.y for point in marker.corners]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    return img[y_min:y_max, x_min:x_max]


def rate_marker_sharpness(marker_img: cv.typing.MatLike) -> int:
    gray = cv.cvtColor(marker_img, cv.COLOR_BGR2GRAY)
    laplacian = cv.Laplacian(gray, cv.CV_64F)
    sharpness = int(laplacian.var())
    return sharpness


def rate_image_focus(img: cv.typing.MatLike) -> int:
    marker = find_best_test_marker(img)
    if marker is None:
        return 0
    crop = crop_out_marker(img, marker)
    return rate_marker_sharpness(crop)


def find_best_focus_image_index(imgs: typing.List[cv.typing.MatLike]) -> int:
    return max(
        range(len(imgs)), key=lambda i: rate_image_focus(imgs[i]), default=None
    )


def find_best_focus_image_in_dir(dir: Path) -> Path:
    img_paths = collect_image_paths(dir)
    imgs = collect_images(img_paths)
    index = find_best_focus_image_index(imgs)
    return img_paths[index]


if __name__ == "__main__":
    path = find_best_focus_image_in_dir(Path("test-images"))
    print(f"Image with best focus is: {path.name}")