import os  # noqa: N999

from nudenet import NudeDetector  # noqa: N999


def detect_images(path: str | list[str]) -> dict[dict]:
    """
    If list (list of paths to saingle images) is given, returns dict of detection dicts. Where the key is the image path.
    If single image path is given, returns dict with a single detection.
    If a directory is given, returns dict of detection dicts. Where the key is the image filename.
    """  # noqa: E501
    detector = NudeDetector()
    results = {}
    if isinstance(path, list):
        detections = detector.detect_batch(path)
        for img_path, detection in zip(path, detections, strict=False):
            results[img_path] = detection
    else:
        if os.path.isdir(path):
            for filename in os.listdir(path):
                file_path = os.path.join(path, filename)
                detection = detector.detect(file_path)
                results[filename] = detection
        else:
            results[path] = detector.detect(path)

    return results
