module NeuralNet.Cost (computeCost) where

import Numeric.LinearAlgebra


computeCost :: Matrix Double -> Matrix Double -> Double
computeCost al y
  | rows al /= 1       = error "al should be a row vector"
  | rows y /= 1        = error "y should be a row vector"
  | cols y /= cols al = error "y and al should be same length"
  -- TODO: Must be a better impl than this sum
  | otherwise           = (-1 / fromIntegral m) * sum (concat (toLists (term1 + term2)))
    where
      m = cols y
      unitRowVector = row (replicate m 1)
      term1 = y * cmap log al
      term2 = (unitRowVector - y) * (cmap log (unitRowVector - al))
