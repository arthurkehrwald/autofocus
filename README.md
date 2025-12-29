# Autofocus

This small tool identifies the sharpest image of a Siemens star in a directory of images.

Usage:

```bash
python3 autofocus/auto_focus.py --dir test-images --output-crops crops --verbose
```

Options:
- `--dir`: directory containing images
- `--crop-fraction`: fraction of the smaller image dimension used for the detection crop (default 0.25)
- `--output-crops`: optional folder to save the detected crop for each image
- `--verbose`: print per-image scores

Dependencies: see `requirements.txt` (OpenCV and NumPy).

Example:

```bash
pip install -r requirements.txt
python3 autofocus/auto_focus.py --dir test-images --output-crops detected --verbose
```
