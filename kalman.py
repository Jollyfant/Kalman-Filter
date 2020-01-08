import numpy as np
import matplotlib.pyplot as plt

class KalmanFilter():

  """
  def KalmanFilter
  Applies a specific Kalman filter to time variant observations
  Implemented after: https://en.wikipedia.org/wiki/Kalman_filter#Details
  Assuming moving object with hidden variables x (position), xv (velocity)
  subject to a constant acceleration of 2
  """

  # Assume a constant acceleration of 2 m/s^2
  # And a timestep for each observation of 1 second
  ACCEL = 2
  DELTA = 1

  def __init__(self, stateVectorNaut, processCovarianceNaut):

    """
    def KalmanFilter.__init__
    Initializes initial estimates of the state and covariance
    """

    # Save input names and values
    self.stateVector = np.fromiter(stateVectorNaut.values(), dtype="float")
    self.stateVectorNames = stateVectorNaut.keys()
    self.processCovariance = processCovarianceNaut

    # Save state histories
    self.__stateHistory = list()
    self.__stateCovarianceHistory = list()

  def transformCovariance(self, A, Z):

    """
    def KalmanFilter.transformCovariance
    Transformation rule for covariance matrix
    """

    return A @ Z @ A.T

  def plot(self):

    """
    def KalmanFilter.plot
    Plots the filter history for all parameters
    """

    plt.plot(self.__stateHistory)
    plt.legend(self.stateVectorNames)
    plt.ylabel("Internal State")
    plt.xlabel("Time Step (dt)")
    plt.title("Kalman Filter State Evolution")
    plt.show()

    plt.plot(self.__stateCovarianceHistory)
    plt.legend(self.stateVectorNames)
    plt.ylabel("Model Variance")
    plt.xlabel("Time Step (dt)")
    plt.title("Kalman Filter Variance Evolution")
    plt.show()

  @property
  def controlMatrix(self):

    """
    def KalmanFilter.controlMatrix
    Maps the control to the next predicted state (physics)
    x1 = 0.5 * a * t^2
    v1 = a * t
    """

    return np.array([
      0.5 * self.ACCEL * self.DELTA ** 2,
      self.ACCEL * self.DELTA
    ])

  @property
  def stateMatrix(self):

    """
    def KalmanFilter.stateMatrix
    Maps the internal state to the next predicted state (physics)
    x1 = x0 + v0 * dt
    v1 = v0
    """

    return np.array([
      [1, self.DELTA],
      [0, 1]
    ])

  @property
  def processNoise(self):
    return np.zeros(2)

  @property
  def processNoiseCovariance(self):
    return np.zeros((2, 2))

  @property
  def observationMatrix(self):
    # Utility matrix to make others compatible
    return np.identity(2)

  def predictState(self):

    """
    def KalmanFilter.predict
    Prediction step to estimate the next state (and covariance) based on current state (physics)
    """

    # Prediction step (a priori state estimate & covariance)
    # Transform covariance: AZA'
    return (
      self.stateMatrix @ self.stateVector + self.controlMatrix + self.processNoise,
      self.transformCovariance(self.stateMatrix, self.processCovariance) + self.processNoiseCovariance
    )


  def addObservation(self, measurement, measurementCovariance):

    """
    def KalmanFilter.addObservation
    Observe a new measurement
    """

    # Predict the next step
    statePrediction, processCovariancePrediction = self.predictState()

    # Eliminate covariances outside of diagonals
    processCovariancePrediction = np.diag(np.diag(processCovariancePrediction))

    # Optimal Kalman Gain: relative measure of importance of the new measurement vs state
    gain = self.calculateKalmanGain(
      processCovariancePrediction,
      measurementCovariance
    )

    # Calculate the innovation 
    innovation = measurement - (self.observationMatrix @ statePrediction)

    # Update the a posteriori state and covariance
    self.stateVector = statePrediction + gain @ innovation
    self.processCovariance = (np.identity(len(gain)) - gain @ self.observationMatrix) @ processCovariancePrediction

    # The uost-fit residual
    residual = measurement - (self.observationMatrix @ self.stateVector)

    self.__stateHistory.append(self.stateVector)
    self.__stateCovarianceHistory.append(np.diag(self.processCovariance))

    return self.stateVector

  def calculateKalmanGain(self, processCovariancePrediction, measurementCovariance):

    """
    def KalmanFilter.calculateKalmanGain
    Returns the optimal Kalman gain
    """

    # Calculate the innovation covariance
    innovationCovariance = self.transformCovariance(self.observationMatrix, processCovariancePrediction) + measurementCovariance
    
    # Calculate the optimal Kalman gain
    return processCovariancePrediction @ self.observationMatrix.T @ np.linalg.inv(innovationCovariance)

  @property
  def currentState(self):
    return dict(zip(self.stateVectorNames, self.stateVector))
