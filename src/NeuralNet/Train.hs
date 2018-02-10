module NeuralNet.Train (
  nnBackward
) where

import NeuralNet.Net
import NeuralNet.Example
import NeuralNet.Matrix
import Data.Matrix
import qualified Data.Vector as Vector


nnBackward :: NeuralNet -> [[Double]] -> ExampleSet -> [(Matrix Double, Matrix Double)]
nnBackward nn as examples
  | length (nnLayers nn) /= 1 = error "Not yet impl"
  | otherwise                 = [(dw, db)]
    where
      x = exampleSetX examples
      y = exampleSetY examples
      m = fromIntegral (ncols x)
      dz = colVector (Vector.fromList (head as)) - y
      dw = mapMatrix (/ m) (x * transpose dz)
      db = mapMatrix (/ m) dz
