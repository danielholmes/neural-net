module NeuralNet.Matrix (
  mapMatrix,
  projectMatrixX
) where

import Data.Matrix

mapMatrix :: (a -> a) -> Matrix a -> Matrix a
mapMatrix f m = foldr (mapRow (\_ x -> f x)) m [1..(nrows m)]

projectMatrixX :: Matrix a -> Int -> Matrix a
projectMatrixX m n = foldl (\mx _ -> mx <|> m) m [1..(n - 1)]
