module NeuralNet.Cost (cost) where

import Data.Matrix
import NeuralNet.Matrix


cost :: Matrix Double -> Matrix Double -> Double
cost al y
  | nrows al /= 1       = error "al should be a row vector"
  | nrows y /= 1        = error "y should be a row vector"
  | ncols y /= ncols al = error "y and al should be same length"
  | otherwise           = (-1 / (fromIntegral m)) * sumMatrix (term1 + term2)
    where
      m = ncols y
      unitRowVector = matrix 1 m (const 1)
      term1 = elementwise (*) y (mapMatrix log al)
      term2 = elementwise (*) (unitRowVector - y) (mapMatrix log (unitRowVector - al))
