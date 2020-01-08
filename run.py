import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from kalman import KalmanFilter

if __name__ == "__main__":

  """
  Demo implementation of a Kalman Filter
  For two hidden state variables x (position) and xv (velocity) with a constant acceleration
  """

  # Initial estimates of position and velocity and (co)-variances
  stateNaut = OrderedDict([
    ("position (m)", 4000),
    ("velocity (m/s)", 280)
  ])

  processCovarianceNaut = np.diag([400.0, 25.0])
  
  # Create the Kalman Filter with the initial state
  # The filter implements the physics internally
  KF = KalmanFilter(stateNaut, processCovarianceNaut)
  
  # Add a list of observations to the filter
  observations = np.array([
    [4260, 282],
    [4550, 285],
    [4860, 286],
    [5110, 290]
  ])
  
  # And a covariance on the observations
  observationCovariance = np.diag([625, 36])
  
  for observation in observations:
    KF.addObservation(observation, observationCovariance)
    print(KF.currentState)

  # Add observations as points
  plt.plot(observations, marker="X", label="_nolegend_", linewidth=0)

  KF.plot()
