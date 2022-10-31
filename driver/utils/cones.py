import numpy as np


class PercievedCone:
    """
    Class to encapsulate a single percieved cone
    """

    def __init__(self, position, probability):
        self.position = position
        self.probability = probability


class Cone:
    """
    Class to encapsulate a single cone
    """

    def __init__(self, position, varience):
        self.position = position
        self.var = varience
        self.seen = True

    def merge(self, other):
        """
        This function takes in two means and two squared variance terms,
        and returns updated gaussian parameters.
        """
        # Calculate the new parameters
        x = (other.var*self.position[0] + self.var*other.position[0]) / (self.var + other.var)
        y = (other.var*self.position[1] + self.var*other.position[1]) / (self.var + other.var)
        var = 1 / ((1 / self.var) + (1 / other.var))
        return Cone(np.array([x, y, self.position[2]]), var)

    def distance(self, p):
        """
        Calculate distance to a point (p)
        """
        return np.linalg.norm(p - self.position)

    def angle(self, p, f):
        """
        Calculate angle between a direction vector (f) and the vector pointing from (p) to the cone
        """
        dist = self.distance(p)
        rel = self.position - p
        return np.dot(f, rel) / dist


class ConeCollection:
    """
    Class to encapsulate a set of cones of the same colour
    """

    def __init__(self, cone_radius):
        self._cones = {}
        self.cone_radius = cone_radius

    def _new_key(self):
        """
        Generate new key by incrementing over the current max key
        """
        return max(list(self._cones.keys()) + [0]) + 1

    def _clear_flags(self):
        """
        Set the seen flag on each cone to false
        """
        for i in self._cones:
            self._cones[i].seen = False

    def _merge(self):
        """
        Merge cones that are within cone radius to each other
        """
        to_be_merged = []

        for i, c1 in self._cones.items():
            for j, c2 in self._cones.items():
                if i != j and np.linalg.norm(c2.position - c1.position) < self.cone_radius:
                    to_be_merged.append((i, j))

        for i, j in to_be_merged:
            if i not in self._cones or j not in self._cones:    # Check if already removed
                continue
            c1 = self._cones.pop(i)
            c2 = self._cones.pop(j)
            # Create new merged cone
            self._cones[self._new_key()] = c1.merge(c2)

    def _prune(self, seen_cones, p, f):
        """
        Given the position (p) and forward (f) of the car reduce the weight of cones
        that are in front of the vehicle, close and were not seen.
        Remove the ones that reach negative weight

        This is a dumb approximation of doing a proper Bayesian update
        """
        to_be_removed = []

        for i, c in self._cones.items():
            # Ignore the ones already seen
            if c.seen:
                continue

            angle = c.angle(p, f)
            relative_distance = c.distance(p)

            # Ignore the ones that are not in front of the car
            if angle < 0.6:         # Magic number: Dumb approximation of camera FOV
                continue

            # Ignore cones too far from the vehicle
            if relative_distance > 25:         # Magic number: rough size of the inclusion box
                continue

            # Increase the variance the closer or the more forward it is
            w = 20 / relative_distance * angle

            still_seen = any(
                np.linalg.norm(c.position - Cone(seen_cone[0], 1/seen_cone[1]).position)
                < self.cone_radius + 1 for seen_cone in seen_cones
            )
            if not still_seen and angle < 0.3 and relative_distance > 3:
                self._cones[i].var += w

            self._cones[i].var += w

            if self._cones[i].var >= 50:
                to_be_removed.append(i)     # Mark for removal

        for i in to_be_removed:
            self._cones.pop(i)

    def _add_cone(self, c1):
        """
        Add a new cone (c1), either by updating an existing one, or adding a new one
        """
        c1 = Cone(c1[0], 1/c1[1])
        for i, c2 in self._cones.items():
            if np.linalg.norm(c1.position - c2.position) < self.cone_radius:
                self.remove_cone(c2)
                c1 = c1.merge(c2)

        self._cones[self._new_key()] = c1

    def add_cones(self, cs, p, f):
        """
        Called once per update
        Add all the cones (cs), car position (p) and camera forward (f) is used to prune
        """
        self._clear_flags()
        for c in cs:
            self._add_cone(c)
        self._merge()
        self._prune(cs, p, f)

    def remove_cone(self, c):
        """
        Remove a cone from _cones
        """
        self._cones = {key: val for key, val in self._cones.items() if val != c}

    def external_merge(self, c1, c2):
        """
        Merge a local cone with a cone from another collection
        """
        if c1 not in self._cones.values():
            return
        self.remove_cone(c1)
        self._cones[self._new_key()] = c1.merge(c2)

    @property
    def cones(self):
        return list(self._cones.values())
