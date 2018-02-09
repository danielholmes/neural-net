module NeuralNet.Matrix (
  mapMatrix
) where

import Data.Matrix

mapMatrix :: (a -> a) -> Matrix a -> Matrix a
mapMatrix f m = foldr (mapRow (\_ x -> f x)) m [1..(nrows m)]
