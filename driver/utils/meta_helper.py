from fs_msgs.msg import MetaPoint, MetaKeyValuePair, MetaPointCollection  # MetaData
from fs_utils.conversions import np_to_vector3
import numpy as np


def dict_to_kvps(d):
    """
    Convert a dictionary to a metadat key-value pair
    """
    return [MetaKeyValuePair(key=str(key), value=str(value)) for key, value in d.items()]


def cones_to_point_collection(cones, name, colour, point_type='cone', return_var=True, var_scale=20):
    """
    Converts cone objects with probability and position properties to metadata representation
    """
    def cone_to_point(cone):
        """
        Convert a single cone to a metadata point
        """
        cone_properties = {'varience': cone.var}

        return MetaPoint(position=np_to_vector3(cone.position),
                         properties=dict_to_kvps(cone_properties))

    def var_to_point(cone):
        """
        Convert a single cone to a metadata point
        """
        cone_properties = {'radius': np.clip(cone.var/var_scale, a_min=0.5, a_max=3)}

        return MetaPoint(position=np_to_vector3(cone.position),
                         properties=dict_to_kvps(cone_properties))

    if return_var:
        return (
            MetaPointCollection(
                points=list(map(cone_to_point, cones)),
                properties=dict_to_kvps({'name': name, 'colour': colour, 'type': point_type})),
            MetaPointCollection(
                points=list(map(var_to_point, cones)),
                properties=dict_to_kvps({'name': 'variance', 'colour': colour, 'type': 'circle'}))
        )
    else:
        return (
            MetaPointCollection(
                points=list(map(cone_to_point, cones)),
                properties=dict_to_kvps({'name': name, 'colour': colour, 'type': point_type})),
        )


def percieved_cones_to_point_collection(cones, name, colour, point_type='square'):
    """
    Converts cone objects with probability and position properties to metadata representation
    """
    def cone_to_point(cone):
        """
        Convert a single cone to a metadata point
        """
        cone_properties = {'size': np.clip(cone.probability, a_min=0.3, a_max=0.8)}

        return MetaPoint(position=np_to_vector3(cone.position),
                         properties=dict_to_kvps(cone_properties))

    return MetaPointCollection(
        points=list(map(cone_to_point, cones)),
        properties=dict_to_kvps({'name': name, 'colour': colour, 'type': point_type}))


def points_to_point_collection(points, name, colour, point_type='line'):
    """
    Converts collection of coordinates to metadata representation
    """
    def point_to_meta_point(point):
        """
        Convert a single coordinate point to metadata
        """
        return MetaPoint(position=np_to_vector3(point))

    return MetaPointCollection(
        points=list(map(point_to_meta_point, points)),
        properties=dict_to_kvps({'name': name, 'colour': colour, 'type': point_type}))
