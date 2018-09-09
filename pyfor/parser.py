def parse_crs(las_object):
    """
    Attempts to parse CRS information from a laspy object.

    :param las_object: A `laspy.file.File` object opened in read mode.
    :return:
    """

    header = las_object.header

    # Check las version
    if header.version == '1.4':
        # Get wkt
        return header.vlrs[0].parsed_body[0]
    else:
        pass
