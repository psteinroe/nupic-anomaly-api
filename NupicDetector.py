# ----------------------------------------------------------------------
# Copyright (C) 2014, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------


import math

from nupic.algorithms import anomaly_likelihood
from nupic.frameworks.opf.common_models.cluster_params import (
    getScalarMetricWithTimeOfDayAnomalyParams)

from nupic.frameworks.opf.model_factory import ModelFactory

class NupicDetector(object):
    """
  This detector uses an HTM based anomaly detection technique.
  """

    def __init__(self, inputMin, inputMax, probationaryPeriod, *args, **kwargs):

        self.inputMin = inputMin
        self.inputMax = inputMax
        self.probationaryPeriod = probationaryPeriod

        self.model = None
        self.sensorParams = None
        self.anomalyLikelihood = None
        # Keep track of value range for spatial anomaly detection
        self.minVal = None
        self.maxVal = None

        # Set this to False if you want to get results based on raw scores
        # without using AnomalyLikelihood. This will give worse results, but
        # useful for checking the efficacy of AnomalyLikelihood. You will need
        # to re-optimize the thresholds when running with this setting.
        self.useLikelihood = True

        # Just for later reference
        self.model_params = None

    def handleRecord(self, inputData):
        """Returns a tuple (anomalyScore, rawScore).
    Internally to NuPIC "anomalyScore" corresponds to "likelihood_score"
    and "rawScore" corresponds to "anomaly_score". Sorry about that.
    """
        # Send it to Numenta detector and get back the results
        result = self.model.run(inputData)

        # Retrieve the anomaly score and write it to a file
        rawScore = result.inferences["anomalyScore"]

        if self.useLikelihood:
            # Compute log(anomaly likelihood)
            anomalyScore = self.anomalyLikelihood.anomalyProbability(
                inputData["value"], rawScore, inputData["timestamp"])
            logScore = self.anomalyLikelihood.computeLogLikelihood(anomalyScore)
            finalScore = logScore
        else:
            finalScore = rawScore

        return finalScore, rawScore

    def get_model_params(self):
        return self.model_params

    def initialize(self):
        # Get config params, setting the RDSE resolution
        rangePadding = abs(self.inputMax - self.inputMin) * 0.2
        self.model_params = getScalarMetricWithTimeOfDayAnomalyParams(
            metricData=[0],
            minVal=self.inputMin - rangePadding,
            maxVal=self.inputMax + rangePadding,
            minResolution=0.001,
            tmImplementation="cpp"
        )["modelConfig"]

        self._setupEncoderParams(
            self.model_params["modelParams"]["sensorParams"]["encoders"])

        self.model = ModelFactory.create(self.model_params)

        self.model.enableInference({"predictedField": "value"})

        if self.useLikelihood:
            # Initialize the anomaly likelihood object
            learningPeriod = int(math.floor(self.probationaryPeriod / 2.0))
            self.anomalyLikelihood = anomaly_likelihood.AnomalyLikelihood(
                learningPeriod=learningPeriod,
                estimationSamples=self.probationaryPeriod - learningPeriod,
                reestimationPeriod=1
            )

    def _setupEncoderParams(self, encoderParams):
        # The encoder must expect the NAB-specific datafile headers
        encoderParams["timestamp_dayOfWeek"] = encoderParams.pop("c0_dayOfWeek")
        encoderParams["timestamp_timeOfDay"] = encoderParams.pop("c0_timeOfDay")
        encoderParams["timestamp_timeOfDay"]["fieldname"] = "timestamp"
        encoderParams["timestamp_timeOfDay"]["name"] = "timestamp"
        encoderParams["timestamp_weekend"] = encoderParams.pop("c0_weekend")
        encoderParams["value"] = encoderParams.pop("c1")
        encoderParams["value"]["fieldname"] = "value"
        encoderParams["value"]["name"] = "value"

        self.sensorParams = encoderParams["value"]
