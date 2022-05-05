from utils import net_scope


class Sensor:
    """
        Sensor class definition. It contains all the useful data for the proper management of the algorithm.

        Parameters
        ----------
        sensor : str
            The name of the sensor which has provided the image.
    """

    def __init__(self, sensor):

        self.sensor = sensor
        self.ratio = 4

        if (sensor == 'QB') or (sensor == 'GeoEye1') or (sensor == 'GE1') or (sensor == 'WV2') or (sensor == 'WV3'):
            self.kernels = [9, 5, 5]
        elif (sensor == 'Ikonos') or (sensor == 'IKONOS'):
            self.kernels = [5, 5, 5]

        if (sensor == 'QB') or (sensor == 'GeoEye1') or (sensor == 'GE1') or (sensor == 'Ikonos') or (
                sensor == 'IKONOS'):
            self.nbands = 4
        elif (sensor == 'WV2') or (sensor == 'WV3'):
            self.nbands = 8
        self.net_scope = net_scope(self.kernels)
        self.nbits = 11

        if sensor == 'WV2' or sensor == 'WV3':
            self.beta = 0.36
            self.learning_rate = 1e-5
        elif (sensor == 'GE1') or (sensor == 'GeoEye1'):
            self.beta = 0.25
            self.learning_rate = 5 * 1e-5
