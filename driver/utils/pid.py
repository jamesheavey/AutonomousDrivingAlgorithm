
class PIDController:
    """
    Class to hold simple PID controller
    """

    def __init__(self, P_value=0, I_value=0, D_value=0, time=None):
        self._P = P_value
        self._I = I_value
        self._D = D_value
        self._time = time or 0
        self._dE = None
        self._iE = 0
        self.last_value = None

    def update(self, e, time):
        """
        Update the PID controller with current error (e) and current time
        Use these, to calculate and return control value a
        """
        # If time didn't chagne since last update, return last value
        if time == self._time:
            if self.last_value is not None:
                return self.last_value
            # If no last value available, return 0
            return 0

        dt = time - self._time  # Calculate time since last update
        self._time = time   # Save time for next update

        # Calculate the change in error since last update
        dE = (e - self._dE) / dt if self._dE is not None else 0
        # Save current error value for the next update
        self._dE = e

        # Calculate the cumulative error
        iE = self._iE + e * dt
        # Save the cumulative error for the next update
        self._iE = iE

        # Calculate control value
        a = e * self._P + dE * self._D + iE * self._I
        self.last_value = a
        return a
