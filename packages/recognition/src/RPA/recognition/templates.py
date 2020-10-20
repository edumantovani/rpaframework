import logging
import time
from pathlib import Path
from typing import Any, Iterator, List, Optional, Union

import cv2
import numpy
from PIL import Image
from RPA.core import geometry
from RPA.core.geometry import Region


DEFAULT_CONFIDENCE = 0.95


class ImageNotFoundError(Exception):
    """Raised when template matching fails."""


def find(
    image: Union[Image.Image, Path],
    template: Union[Image.Image, Path],
    region: Optional[Region] = None,
    limit: Optional[int] = None,
    confidence: float = DEFAULT_CONFIDENCE,
) -> List[Region]:
    """Attempt to find the template from the given image.

    :param image:       Path to image or Image instance, used to search from
    :param template:    Path to image or Image instance, used to search with
    :param limit:       Limit returned results to maximum of `limit`.
    :param region:      Area to search from. Can speed up search significantly.
    :param confidence:  Confidence for matching, value between 0.1 and 1.0
    :return:            List of matching regions
    :raises ImageNotFoundError: No match was found
    """
    # Ensure images are in Pillow format
    image = _to_image(image)
    template = _to_image(template)

    # Crop image if requested
    if region is not None:
        region = geometry.to_region(region)
        image = image.crop(region.as_tuple())

    # Verify template still fits in image
    if template.size[0] > image.size[0] or template.size[1] > image.size[1]:
        raise ValueError("Template is larger than search region")

    # Do the actual search
    start = time.time()

    matches = []
    for match in _match_template(image, template, confidence):
        matches.append(match)
        if limit is not None and len(matches) >= int(limit):
            break

    logging.info("Scanned image in %.2f seconds", time.time() - start)

    if not matches:
        raise ImageNotFoundError("No matches for given template")

    # Convert region coördinates back to full-size coördinates
    if region is not None:
        for match in matches:
            match.move(region.left, region.top)

    return matches


def _to_image(obj: Any) -> Image.Image:
    """Convert `obj` to instance of Pillow's Image class."""
    if obj is None or isinstance(obj, Image.Image):
        return obj
    return Image.open(obj)


def _match_template(
    image: Image.Image, template: Image.Image, confidence: float = DEFAULT_CONFIDENCE
) -> Iterator[Region]:
    """Use opencv's matchTemplate() to slide the `template` over
    `image` to calculate correlation coefficients, and then
    filter with a confidence to find all relevant global maximums.
    """
    confidence = max(0.01, min(confidence, 1.00))
    template_width, template_height = template.size

    if image.mode == "RGBA":
        image = image.convert("RGB")
    if template.mode == "RGBA":
        template = template.convert("RGB")

    image = numpy.array(image)
    template = numpy.array(template)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    template = cv2.cvtColor(template, cv2.COLOR_RGB2BGR)

    coefficients = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)

    while True:
        _, match_coeff, _, (match_x, match_y) = cv2.minMaxLoc(coefficients)
        if match_coeff < confidence:
            break

        coefficients[
            match_y - template_height // 2 : match_y + template_height // 2,
            match_x - template_width // 2 : match_x + template_width // 2,
        ] = 0

        yield Region.from_size(match_x, match_y, template_width, template_height)
